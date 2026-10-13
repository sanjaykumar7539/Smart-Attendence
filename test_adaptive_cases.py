import numpy as np
from types import SimpleNamespace
from adaptive_recognition import AdaptiveRecognitionManager


def make_cfg():
    return SimpleNamespace(
        good_face_px=96, medium_face_px=60,
        blur_good=90.0, blur_min=35.0,
        brightness_min=45.0, brightness_max=215.0,
        face_buffer_size=5, recognition_vote_window=5, recognition_vote_n=3,
        sr_trigger_face_px=60, sr_max_calls_per_track=1,
        threshold_good=0.45, threshold_medium=0.40, threshold_poor=0.36,
        max_search_k=8, adaface_threshold=0.40, adaface_weight=0.35,
        enable_super_resolution=False, sr_model_path='./models/missing_sr.onnx',
        enable_adaface=False, adaface_model_path='./models/missing_ada.onnx',
        adaface_index_path='./database/missing_ada.faiss',
        adaface_meta_path='./database/missing_ada.json'
    )

class FakeEngine:
    def __init__(self, seq):
        self.seq=list(seq); self.i=0
    def identify_detailed(self, emb, k=8, threshold=0.0):
        item=self.seq[min(self.i, len(self.seq)-1)]
        self.i += 1
        eid, score, margin=item
        if eid is not None and score >= threshold and margin >= 0.06:
            return eid, score, margin
        return None, score, margin
    def extract_embedding_from_crop(self, crop):
        return np.ones(512, dtype=np.float32)

class FakeSR:
    available=True
    def enhance(self, crop):
        return np.repeat(np.repeat(crop, 2, axis=0), 2, axis=1)

class FakeAda:
    def __init__(self, identity='S1', score=.70):
        self.available=True; self.identity=identity; self.score=score
    def embed(self,crop): return np.ones(512,dtype=np.float32)
    def identify(self,emb,k=8): return self.identity,self.score,.20


def frame_and_face(side=120, noisy=True):
    frame=np.full((240,240,3),128,dtype=np.uint8)
    x1,y1=30,30; x2,y2=x1+side,y1+side
    if noisy:
        rng=np.random.default_rng(7)
        frame[y1:y2,x1:x2]=rng.integers(40,216,size=(side,side,3),dtype=np.uint8)
    face={'bbox':[x1,y1,x2,y2], 'embedding':np.ones(512,dtype=np.float32), 'landmark':None}
    return frame,face


def test_confirmed_after_votes():
    mgr=AdaptiveRecognitionManager(FakeEngine([('S1',.70,.20)]*3), make_cfg())
    f,face=frame_and_face(120)
    sts=[mgr.update(1,f,face).status for _ in range(3)]
    assert sts == ['UNCERTAIN','UNCERTAIN','CONFIRMED'], sts


def test_unknown_low_margin():
    mgr=AdaptiveRecognitionManager(FakeEngine([('S1',.70,.02)]), make_cfg())
    f,face=frame_and_face(120)
    d=mgr.update(2,f,face)
    assert d.status=='UNKNOWN' and d.employee_id is None, d


def test_uncertain_before_vote_quorum():
    mgr=AdaptiveRecognitionManager(FakeEngine([('S2',.65,.12)]), make_cfg())
    f,face=frame_and_face(120)
    d=mgr.update(3,f,face)
    assert d.status=='UNCERTAIN' and d.employee_id=='S2', d


def test_adaface_conflict_blocks_identity():
    mgr=AdaptiveRecognitionManager(FakeEngine([('S1',.70,.20)]), make_cfg())
    mgr.ada=FakeAda(identity='S9',score=.80)
    f,face=frame_and_face(120)
    d=mgr.update(4,f,face)
    assert d.status=='UNKNOWN' and d.employee_id is None, d


def test_superres_only_once_per_track():
    cfg=make_cfg(); cfg.sr_trigger_face_px=60; cfg.sr_max_calls_per_track=1
    mgr=AdaptiveRecognitionManager(FakeEngine([('S1',.70,.20)]*3), cfg)
    mgr.sr=FakeSR()
    f,face=frame_and_face(40)
    mgr.update(5,f,face); mgr.update(5,f,face); mgr.update(5,f,face)
    assert mgr.sr_calls[5] == 1, mgr.sr_calls[5]


def test_cleanup_removes_old_tracks():
    mgr=AdaptiveRecognitionManager(FakeEngine([('S1',.70,.20)]), make_cfg())
    f,face=frame_and_face(120)
    mgr.update(10,f,face); mgr.update(11,f,face)
    mgr.cleanup([11])
    assert 10 not in mgr.buffers and 10 not in mgr.votes and 11 in mgr.buffers


def run_all():
    tests=[
        test_confirmed_after_votes,
        test_unknown_low_margin,
        test_uncertain_before_vote_quorum,
        test_adaface_conflict_blocks_identity,
        test_superres_only_once_per_track,
        test_cleanup_removes_old_tracks,
    ]
    for t in tests:
        t(); print('PASS',t.__name__)
    print(f'TOTAL: {len(tests)}/{len(tests)} PASS')

if __name__=='__main__': run_all()
