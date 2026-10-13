"""Dependency-light smoke tests for quality and adaptive decision logic.
Run: python test_adaptive_pipeline.py
"""
import numpy as np
from types import SimpleNamespace
from quality import FaceQualityAnalyzer


def cfg():
    return SimpleNamespace(good_face_px=96, medium_face_px=60, blur_good=90., blur_min=35., brightness_min=45., brightness_max=215.)


def test_quality_invalid():
    q=FaceQualityAnalyzer(cfg()).analyze(np.zeros((100,100,3),dtype=np.uint8), [20,20,20,40])
    assert q.level=='poor' and q.min_side==0


def test_quality_small_face_is_not_good():
    frame=np.full((200,200,3),128,dtype=np.uint8)
    q=FaceQualityAnalyzer(cfg()).analyze(frame,[20,20,60,60])
    assert q.level!='good'


if __name__=='__main__':
    test_quality_invalid(); test_quality_small_face_is_not_good()
    print('adaptive smoke tests: PASS')
