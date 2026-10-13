"""180 deterministic synthetic robustness cases; no real biometric accuracy claims."""
import json, sys, time
from pathlib import Path
import numpy as np
import cv2
from test_adaptive_cases import make_cfg, FakeEngine, FakeSR, FakeAda
from adaptive_recognition import AdaptiveRecognitionManager

ROOT=Path(__file__).resolve().parent
rng=np.random.default_rng(20260909)
results=[]

def check(name, condition, detail=''):
    results.append({'case':name,'passed':bool(condition),'detail':detail})

def sample(side, angle, blur, light):
    image=np.full((240,240,3),128,np.uint8)
    patch=rng.integers(40,216,(side,side,3),dtype=np.uint8)
    matrix=cv2.getRotationMatrix2D((side/2,side/2),angle,1)
    patch=cv2.warpAffine(patch,matrix,(side,side),borderMode=cv2.BORDER_REFLECT)
    if blur: patch=cv2.GaussianBlur(patch,(0,0),blur)
    patch=np.clip(patch.astype(np.float32)*light,0,255).astype(np.uint8)
    image[20:20+side,20:20+side]=patch
    face={'bbox':[20,20,20+side,20+side],'embedding':np.ones(512,np.float32),'landmark':None}
    return image,face

# 120 image-condition cases: 5 angles x 3 sizes x 2 blur x 4 lighting.
for angle in [-60,-30,0,30,60]:
 for side in [32,64,112]:
  for blur in [0,3]:
   for light in [.12,.35,.7,1.0]:
    f,face=sample(side,angle,blur,light)
    cfg=make_cfg(); cfg.recognition_vote_n=3
    mgr=AdaptiveRecognitionManager(FakeEngine([('S1',.8,.2)]*3),cfg)
    q=mgr.quality.analyze(f,face['bbox'],face.get('landmark'))
    d=mgr.update(1,f,face)
    check(f'image_{angle}_{side}_{blur}_{light}',np.isfinite(q.score) and 0<=q.score<=1 and d.status=='UNCERTAIN' and d.employee_id=='S1',f'quality={q.level}; brightness={q.brightness:.1f}; pose is image rotation, not real 3D yaw')

# 60 decision/safety cases, including conflicting identities and SR fallback.
for i in range(60):
 cfg=make_cfg(); cfg.recognition_vote_n=3
 f,face=sample(40 if i%2 else 112,0,0,1)
 if i%6==0:
  seq=[('S1',.8,.2)]*3; expected='CONFIRMED'
 elif i%6==1:
  seq=[('S1',.8,.01)]*3; expected='UNKNOWN'
 elif i%6==2:
  seq=[('S1',.8,.2),('S2',.8,.2),('S1',.8,.2)]; expected='UNCERTAIN'
 elif i%6==3:
  seq=[('S1',.8,.2)]*3; expected='UNKNOWN'
 elif i%6==4:
  seq=[('S1',.8,.2)]*3; expected='CONFIRMED'
 else:
  seq=[(None,.1,0)]*3; expected='UNKNOWN'
 mgr=AdaptiveRecognitionManager(FakeEngine(seq),cfg)
 if i%6==3: mgr.ada=FakeAda('S9',.8)
 if i%6==4: mgr.sr=FakeSR()
 ds=[mgr.update(i,f,face) for _ in range(3)]
 d=ds[-1]
 ok=d.status==expected
 if i%6==4: ok=ok and mgr.sr_calls[i]==(1 if i%2 else 0)
 if expected=='UNKNOWN': ok=ok and d.employee_id is None
 check(f'decision_{i:03d}',ok,f'expected={expected}; actual={d.status}; SR calls={mgr.sr_calls[i]}')

report={'scope':'Synthetic image transformations and mocked recognition decisions; not real face recognition accuracy','total':len(results),'passed':sum(r['passed'] for r in results),'failed':[r for r in results if not r['passed']],'cases':results}
(ROOT/'TEST_REPORT_180.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(f"RESULT: {report['passed']}/{report['total']} PASS")
for r in report['failed'][:15]: print('FAIL',r)
if report['failed']: sys.exit(1)
