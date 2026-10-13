"""Track-aware adaptive recognition: quality gate → optional SR → ArcFace/AdaFace → voting."""
from collections import defaultdict, deque, Counter
from dataclasses import dataclass
import cv2
import numpy as np
from loguru import logger
from quality import FaceQualityAnalyzer
from super_resolution import SuperResolutionEngine
from adaface_verifier import AdaFaceVerifier


@dataclass
class Decision:
    status:str='UNKNOWN'; employee_id:str|None=None; confidence:float=0.0; quality:str='poor'; enhanced:bool=False


class AdaptiveRecognitionManager:
    def __init__(self, engine, cfg):
        self.engine=engine; self.cfg=cfg
        self.quality=FaceQualityAnalyzer(cfg)
        self.sr=SuperResolutionEngine(cfg)
        self.ada=AdaFaceVerifier(cfg)
        self.buffers=defaultdict(lambda:deque(maxlen=cfg.face_buffer_size))
        self.votes=defaultdict(lambda:deque(maxlen=cfg.recognition_vote_window))
        self.sr_calls=defaultdict(int)

    @staticmethod
    def _crop(frame,bbox,pad=.15):
        h,w=frame.shape[:2]; x1,y1,x2,y2=map(int,bbox); bw=x2-x1; bh=y2-y1
        x1=max(0,int(x1-bw*pad)); y1=max(0,int(y1-bh*pad)); x2=min(w,int(x2+bw*pad)); y2=min(h,int(y2+bh*pad))
        return frame[y1:y2,x1:x2].copy()

    def update(self,track_id,frame,face):
        q=self.quality.analyze(frame,face['bbox'],face.get('landmark'))
        crop=self._crop(frame,face['bbox'])
        self.buffers[track_id].append((q.score,crop,q,face))
        # Prefer the best observed frame for medium/poor tracks; current frame for good faces.
        sample=max(self.buffers[track_id],key=lambda x:x[0]) if q.level!='good' else self.buffers[track_id][-1]
        _,best_crop,bq,bface=sample
        enhanced=False; arc_emb=bface.get('embedding')
        if bq.min_side <= self.cfg.sr_trigger_face_px and self.sr.available and self.sr_calls[track_id] < self.cfg.sr_max_calls_per_track:
            sr_crop=self.sr.enhance(best_crop); self.sr_calls[track_id]+=1
            sr_emb=self.engine.extract_embedding_from_crop(sr_crop)
            if sr_emb is not None: arc_emb=sr_emb; best_crop=sr_crop; enhanced=True

        threshold={'good':self.cfg.threshold_good,'medium':self.cfg.threshold_medium,'poor':self.cfg.threshold_poor}[bq.level]
        arc_id,arc_score,arc_margin=self.engine.identify_detailed(arc_emb,k=self.cfg.max_search_k,threshold=threshold)
        final_id=arc_id; final_score=arc_score

        # AdaFace is a verifier: it can strengthen a matching ArcFace candidate, never override with a conflicting ID.
        if arc_id and self.ada.available:
            aemb=self.ada.embed(best_crop); aid,ascore,amargin=self.ada.identify(aemb,k=self.cfg.max_search_k)
            if aid==arc_id and ascore>=self.cfg.adaface_threshold:
                w=float(self.cfg.adaface_weight); final_score=(1-w)*arc_score+w*ascore
            elif aid and aid!=arc_id:
                final_id=None  # disagreement = uncertain, do not guess

        self.votes[track_id].append(final_id or '__UNKNOWN__')
        counts=Counter(self.votes[track_id])
        if final_id and counts[final_id] >= self.cfg.recognition_vote_n:
            return Decision('CONFIRMED',final_id,final_score,bq.level,enhanced)
        if final_id:
            return Decision('UNCERTAIN',final_id,final_score,bq.level,enhanced)
        return Decision('UNKNOWN',None,final_score,bq.level,enhanced)

    def cleanup(self,active_ids):
        active=set(active_ids)
        for store in (self.buffers,self.votes,self.sr_calls):
            for k in list(store.keys()):
                if k not in active: store.pop(k,None)
