"""Optional AdaFace ONNX verifier with its own FAISS gallery."""
from pathlib import Path
import json
import cv2
try:
    import faiss
except Exception:
    faiss = None
import numpy as np
from loguru import logger
try:
    import onnxruntime as ort
except Exception:
    ort=None


class AdaFaceVerifier:
    def __init__(self,cfg):
        self.cfg=cfg; self.session=None; self.index=None; self.meta=[]
        mp,ip,jp=Path(cfg.adaface_model_path),Path(cfg.adaface_index_path),Path(cfg.adaface_meta_path)
        if not cfg.enable_adaface: return
        if ort is None or faiss is None or not (mp.exists() and ip.exists() and jp.exists()):
            logger.warning('AdaFace verifier unavailable — ArcFace-only mode')
            return
        try:
            self.session=ort.InferenceSession(str(mp),providers=['CPUExecutionProvider'])
            self.index=faiss.read_index(str(ip)); self.meta=json.loads(jp.read_text(encoding='utf-8'))
            if self.index.ntotal != len(self.meta): raise ValueError('AdaFace index/meta length mismatch')
            logger.success(f'AdaFace verifier ready: {self.index.ntotal} templates')
        except Exception as e:
            logger.warning(f'AdaFace disabled: {e}'); self.session=None; self.index=None; self.meta=[]

    @property
    def available(self): return self.session is not None and self.index is not None and self.index.ntotal>0

    def embed(self,face_bgr):
        if not self.available or face_bgr is None or face_bgr.size==0: return None
        img=cv2.resize(face_bgr,(112,112)); img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB).astype(np.float32)
        img=(img/127.5)-1.0; x=np.transpose(img,(2,0,1))[None]
        try:
            inp=self.session.get_inputs()[0].name
            y=self.session.run(None,{inp:x})[0].reshape(-1).astype(np.float32)
            n=np.linalg.norm(y); return y/n if n>1e-8 else None
        except Exception as e:
            logger.debug(f'AdaFace inference failed: {e}'); return None

    def identify(self,emb,k=8):
        if emb is None or not self.available: return None,0.0,0.0
        k=min(k,self.index.ntotal); sims,inds=self.index.search(emb.reshape(1,-1).astype(np.float32),k)
        by={}
        for s,i in zip(sims[0],inds[0]):
            if 0<=i<len(self.meta): by[self.meta[i]]=max(by.get(self.meta[i],-1),float(s))
        ranked=sorted(by.items(),key=lambda x:x[1],reverse=True)
        if not ranked:return None,0.0,0.0
        top,score=ranked[0]; second=ranked[1][1] if len(ranked)>1 else -1.0
        return top,score,score-second
