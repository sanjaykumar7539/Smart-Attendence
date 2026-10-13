import json
from collections import Counter
from pathlib import Path
import numpy as np

RESULTS = Path("student_index/phase5_4_raw_recognition_2fps.json")
STUDENTS = Path("student_index/students.json")

with open(RESULTS, encoding="utf-8") as f:
    results = json.load(f)

with open(STUDENTS, encoding="utf-8") as f:
    students = json.load(f)

enrolled = sorted(set(x["student_name"] for x in students))

print("=" * 78)
print("PHASE 5.5 — RAW RECOGNITION AUDIT")
print("=" * 78)

print(f"Videos                 : {len(results['videos'])}")
print(f"Enrolled CSBS students : {len(enrolled)}")
print(f"FAISS vectors           : {results['faiss_vectors']}")
print(f"Embedding dimension     : {results['faiss_dimension']}")
print(f"Sampling FPS            : {results['target_sampling_fps']}")
print(f"Temporal voting         : {results['temporal_voting']}")
print()

all_assignments = []

for video in results["videos"]:

    counts = []
    identities = []
    similarities = []

    for frame in video["frames"]:
        counts.append(frame["face_count"])

        for face in frame["faces"]:
            identities.append(face["student_name"])
            similarities.append(float(face["similarity"]))
            all_assignments.append(face)

    print("-" * 78)
    print(video["video"])
    print("-" * 78)

    print(f"Sampled frames          : {video['sampled_frames']}")
    print(f"Detected faces          : {video['detected_faces_total']}")

    print(
        f"Faces/frame             : "
        f"min={min(counts)}, "
        f"median={np.median(counts):.1f}, "
        f"max={max(counts)}, "
        f"average={np.mean(counts):.2f}"
    )

    print(f"Frames with 0 faces     : {sum(x == 0 for x in counts)}")
    print(f"Frames with >=5 faces   : {sum(x >= 5 for x in counts)}")

    unique_ids = set(identities)

    print(f"Unique identities seen  : {len(unique_ids)} / {len(enrolled)}")

    if similarities:
        print(
            f"Similarity              : "
            f"min={min(similarities):.3f}, "
            f"median={np.median(similarities):.3f}, "
            f"max={max(similarities):.3f}"
        )

    print("Top 10 assigned identities:")

    for name, count in Counter(identities).most_common(10):
        print(f"  {name:25s} : {count}")

print()
print("=" * 78)
print("GLOBAL AUDIT")
print("=" * 78)

print(f"Total detected faces     : {len(all_assignments)}")

global_ids = set(x["student_name"] for x in all_assignments)

print(f"Unique identities seen   : {len(global_ids)} / {len(enrolled)}")

missing = sorted(set(enrolled) - global_ids)

print(f"Students never assigned  : {len(missing)}")

if missing:
    print()
    print("Students never assigned:")
    for name in missing:
        print(f"  {name}")

sims = np.array(
    [float(x["similarity"]) for x in all_assignments],
    dtype=float
)

if len(sims):
    print()
    print("Similarity distribution:")
    print(f"  < 0.20       : {np.sum(sims < 0.20)}")
    print(f"  0.20–0.30    : {np.sum((sims >= 0.20) & (sims < 0.30))}")
    print(f"  0.30–0.40    : {np.sum((sims >= 0.30) & (sims < 0.40))}")
    print(f"  0.40–0.50    : {np.sum((sims >= 0.40) & (sims < 0.50))}")
    print(f"  >= 0.50      : {np.sum(sims >= 0.50)}")

print()
print("=" * 78)
print("PHASE 5.5 AUDIT COMPLETE")
print("=" * 78)
