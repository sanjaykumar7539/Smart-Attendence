"""
FaceTrack Pro — enroll.py
Faculty face enrollment into FAISS index.

Usage examples:
    python enroll.py --demo                                    # add 5 demo faculty (placeholder embeddings)
    python enroll.py --id FAC001 --name "Dr. Ramesh" --dept "Computer Science" --category Teaching --images img1.jpg img2.jpg img3.jpg
    python enroll.py --id EMP010 --name "Mr. Kumar" --dept "Admin Office" --category Non-Teaching --live
    python enroll.py --id FAC001 --name "Dr. Ramesh" --dept "Computer Science" --category Teaching --hlc --images img1.jpg img2.jpg
    python enroll.py --batch ./faces                          # batch from folder structure
    python enroll.py --summary                                # print enrollment status

Fields captured per person:
    --id        Employee ID
    --name      Full name
    --dept      Department
    --desig     Designation (e.g. Professor, Lecturer, Office Assistant)
    --category  Teaching | Non-Teaching  (default: Teaching)
    --hlc       Flag — also registers the person in the HLC team DB (hlc_team.db)

Enrollment photos are saved automatically under:
    ./faces/{employee_id}_{name}/img_01.jpg, img_02.jpg, ...
one folder per person, for every enrollment method (--images, --live, --batch).
"""

import os
import cv2
import numpy as np
import json
import argparse
import faiss
from pathlib import Path
from typing import Optional, List
from loguru import logger
from insightface.app import FaceAnalysis

from config import config
from database import db
from quality import enrollment_quality_weight


class FaceEnroller:

    def __init__(self):
        self.app   = None
        self.index = None
        self.meta  = []
        self.template_weights = []
        self._load_model()
        self._load_or_create_index()

    @staticmethod
    def _weights_path() -> Path:
        return Path(config.database.meta_path).with_suffix(".quality.json")

    def _load_model(self):
        logger.info(f"Loading {config.model.model_pack} for enrollment...")
        ctx_id   = config.model.gpu_id if config.model.use_gpu else -1
        self.app = FaceAnalysis(
            name=config.model.model_pack,
            root=config.model.models_dir,
            allowed_modules=["detection", "recognition"],
        )
        # Use larger det_size for enrollment — more accurate on still images
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        logger.success("Enrollment model ready")

    def _load_or_create_index(self):
        idx_path  = config.database.index_path
        meta_path = config.database.meta_path
        os.makedirs(os.path.dirname(idx_path), exist_ok=True)

        if os.path.exists(idx_path) and os.path.exists(meta_path):
            self.index = faiss.read_index(idx_path)
            with open(meta_path, encoding="utf-8") as f:
                self.meta = json.load(f)
            weights_path = self._weights_path()
            if weights_path.exists():
                self.template_weights = json.loads(weights_path.read_text(encoding="utf-8"))
            if len(self.template_weights) != len(self.meta):
                logger.warning("Enrollment quality weights missing or misaligned; using neutral weights")
                self.template_weights = [1.0] * len(self.meta)
            logger.info(f"Loaded index: {self.index.ntotal} embeddings")
        else:
            self.index = faiss.IndexFlatIP(512)
            self.meta  = []
            self.template_weights = []
            logger.info("Created fresh FAISS index")

    def _save(self):
        faiss.write_index(self.index, config.database.index_path)
        with open(config.database.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, indent=2)
        self._weights_path().write_text(
            json.dumps(self.template_weights, indent=2),
            encoding="utf-8",
        )
        logger.info(f"Saved FAISS index: {self.index.ntotal} total embeddings")

    def get_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract ArcFace embedding from a single image. Returns None if no face found."""
        faces = self.app.get(image)
        if not faces:
            return None
        # Pick the largest face in the image
        face = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )
        w = face.bbox[2] - face.bbox[0]
        h = face.bbox[3] - face.bbox[1]
        if min(w, h) < config.model.min_face_size:
            return None
        return face.normed_embedding.astype(np.float32)

    def _save_person_images(self, employee_id: str, name: str, images: list) -> str:
        """
        Save the raw enrollment photos into a dedicated per-person folder:
            {faces_dir}/{employee_id}_{sanitized_name}/img_01.jpg, img_02.jpg, ...
        Accepts a mix of file-path strings and np.ndarray BGR frames.
        Returns the folder path.
        """
        safe_name = "".join(c if c.isalnum() else "_" for c in name).strip("_")
        folder    = os.path.join(config.database.faces_dir, f"{employee_id}_{safe_name}")
        os.makedirs(folder, exist_ok=True)

        # Number sequentially from any images already saved there (safe on re-enrollment)
        existing = len([f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png"))])
        saved    = 0
        for img in images:
            frame = cv2.imread(img) if isinstance(img, str) else img
            if frame is None:
                continue
            existing += 1
            saved    += 1
            out_path  = os.path.join(folder, f"img_{existing:02d}.jpg")
            cv2.imwrite(out_path, frame)

        logger.info(f"Saved {saved} photo(s) to {folder}")
        return folder

    def enroll_faculty(self, employee_id: str, name: str,
                       department: str = "", designation: str = "",
                       staff_category: str = "Teaching", is_hlc: bool = False,
                       images: list = None) -> bool:
        """
        Enroll one faculty member.
        images: list of np.ndarray BGR frames OR file path strings.
        """
        db.add_faculty(employee_id, name, department, designation,
                       staff_category=staff_category, is_hlc=is_hlc)

        if is_hlc:
            from hlc_team_db import HLCTeamDB
            from config import config as _cfg
            HLCTeamDB(_cfg.hlc.team_db_path).add_employee(
                employee_id, name, dept=department, role=designation)
            logger.info(f"Also registered {employee_id} in HLC team DB")

        if not images:
            logger.warning(f"No images provided for {employee_id} — skipped")
            return False

        self._save_person_images(employee_id, name, images)

        embeddings = []
        quality_weights = []
        for img in images:
            if isinstance(img, str):
                img = cv2.imread(img)
            if img is None:
                continue
            faces = self.app.get(img)
            if not faces:
                logger.debug("No face found in one image — skipped")
                continue
            face = max(
                faces,
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            )
            if min(face.bbox[2] - face.bbox[0], face.bbox[3] - face.bbox[1]) < config.model.min_face_size:
                logger.debug("Face below minimum enrollment size — skipped")
                continue
            emb = getattr(face, "normed_embedding", None)
            if emb is None:
                logger.debug("Face has no embedding — skipped")
                continue
            embeddings.append(emb.astype(np.float32))
            quality_weights.append(enrollment_quality_weight(img, face))

        if not embeddings:
            logger.error(f"No valid face embeddings extracted for {employee_id}")
            return False

        faiss_idx = self.index.ntotal
        vectors = np.asarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors /= np.maximum(norms, 1e-12)
        self.index.add(vectors)
        self.meta.extend([employee_id] * len(embeddings))
        self.template_weights.extend(quality_weights)
        db.set_embedding_index(employee_id, faiss_idx)
        self._save()

        logger.success(
            f"Enrolled: {name} ({employee_id}) | "
            f"{len(embeddings)} image(s) | FAISS idx={faiss_idx}"
        )
        return True

    def enroll_from_folder(self, folder_path: str,
                           default_category: str = "Teaching",
                           default_hlc: bool = False):
        """
        Batch enroll from a folder structure:
            enroll_batch/
              roster.csv                       (optional, recommended for mixed batches)
              FAC001_Dr_Ramesh_Kumar/
                img1.jpg
                img2.jpg
              EMP010_Mr_Kumar/
                ...

        roster.csv columns (header row required):
            employee_id,name,department,category,is_hlc

        Example roster.csv:
            employee_id,name,department,category,is_hlc
            FAC001,Dr. Ramesh Kumar,Computer Science,Teaching,yes
            EMP010,Mr. Kumar,Admin Office,Non-Teaching,no

        If roster.csv is present, it sets department/category/HLC per person —
        this is how you enroll a batch with a real mix of staff types.
        If roster.csv is absent, every person in the folder gets the same
        category/HLC (from --category/--hlc on the command line, or the
        defaults: Teaching / not HLC) and department is left blank.
        """
        folder  = Path(folder_path)
        subdirs = [d for d in folder.iterdir() if d.is_dir()]
        logger.info(f"Found {len(subdirs)} faculty folders in {folder_path}")

        roster = {}
        roster_path = folder / "roster.csv"
        if roster_path.exists():
            import csv
            with open(roster_path, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    roster[row["employee_id"].strip()] = row
            logger.info(f"Loaded roster.csv: {len(roster)} entries")
        else:
            logger.info("No roster.csv found — using uniform --category/--hlc for all")

        for fdir in subdirs:
            parts  = fdir.name.split("_", 1)
            emp_id = parts[0]
            name   = parts[1].replace("_", " ") if len(parts) > 1 else fdir.name

            dept, category, is_hlc = "", default_category, default_hlc
            if emp_id in roster:
                r        = roster[emp_id]
                name     = r.get("name", "").strip() or name
                dept     = r.get("department", "").strip()
                category = r.get("category", "").strip() or default_category
                is_hlc   = r.get("is_hlc", "").strip().lower() in ("yes", "y", "1", "true")

            image_paths = list(fdir.glob("*.jpg")) + list(fdir.glob("*.png"))
            images      = [cv2.imread(str(p)) for p in image_paths]
            images      = [img for img in images if img is not None]

            if not images:
                logger.warning(f"No images in {fdir.name} — skipped")
                continue

            self.enroll_faculty(emp_id, name, department=dept,
                                 staff_category=category, is_hlc=is_hlc,
                                 images=images)

        self.print_summary()

    def enroll_live(self, employee_id: str, name: str,
                    department: str = "", num_captures: int = 8,
                    staff_category: str = "Teaching", is_hlc: bool = False):
        """
        Interactive enrollment from the configured video path.
        SPACE = capture frame, Q = finish and enroll.
        """
        logger.info(f"Live enrollment for: {name} ({employee_id})")
        print("\n  SPACE = capture face  |  Q = finish\n")

        video_path = Path(config.camera.video_path)
        cap    = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise OSError(f"Could not open enrollment video: {video_path.resolve()}")
        images = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            display = frame.copy()
            faces   = self.app.get(frame)
            for face in faces:
                x1, y1, x2, y2 = [int(v) for v in face.bbox]
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(
                display,
                f"Captured: {len(images)}/{num_captures}  SPACE=capture  Q=done",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 120), 2,
            )
            cv2.putText(display, name, (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 212, 255), 2)
            cv2.imshow("Enrollment", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" ") and faces:
                images.append(frame.copy())
                print(f"  Captured {len(images)}")
            elif key == ord("q") or len(images) >= num_captures:
                break

        cap.release()
        cv2.destroyAllWindows()

        if images:
            self.enroll_faculty(employee_id, name, department=department,
                                 staff_category=staff_category, is_hlc=is_hlc,
                                 images=images)
        else:
            logger.warning("No images captured — enrollment cancelled")

    def add_demo_faculty(self):
        """Add 5 demo faculty with random placeholder embeddings (for dashboard testing)."""
        logger.warning("Adding DEMO faculty — placeholder embeddings, NOT for production")
        demo = [
            ("FAC001", "Dr. Ramesh Kumar",    "Computer Science",  "Professor"),
            ("FAC002", "Prof. Anita Sharma",  "Mathematics",       "Associate Professor"),
            ("FAC003", "Dr. Vijay Nair",      "Physics",           "Assistant Professor"),
            ("FAC004", "Mrs. Priya Menon",    "English",           "Lecturer"),
            ("FAC005", "Dr. Suresh Babu",     "Chemistry",         "Professor"),
        ]
        for emp_id, name, dept, desig in demo:
            db.add_faculty(emp_id, name, dept, desig)
            emb  = np.random.randn(512).astype(np.float32)
            emb /= np.linalg.norm(emb)
            idx  = self.index.ntotal
            self.index.add(emb.reshape(1, -1))
            self.meta.append(emp_id)
            db.set_embedding_index(emp_id, idx)
            logger.info(f"  Demo enrolled: {name} ({emp_id})")
        self._save()
        logger.success(f"Demo enrollment complete — {len(demo)} faculty added")

    def print_summary(self):
        all_fac = db.get_all_faculty()
        print(f"\n{'─'*95}")
        print(f"{'ID':<10} {'Name':<26} {'Department':<16} {'Category':<14} {'HLC':<5} {'Enrolled'}")
        print(f"{'─'*95}")
        for f in all_fac:
            mark = "✓ YES" if f["face_enrolled"] else "✗ NO"
            hlc  = "Yes" if f.get("is_hlc") else "-"
            cat  = f.get("staff_category", "Teaching")
            print(f"{f['employee_id']:<10} {f['name']:<26} {f['department']:<16} {cat:<14} {hlc:<5} {mark}")
        print(f"{'─'*95}")
        print(f"Total FAISS embeddings: {self.index.ntotal}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FaceTrack Pro — Faculty Enrollment")
    parser.add_argument("--id",      help="Employee ID (e.g. FAC001)")
    parser.add_argument("--name",    help="Full name")
    parser.add_argument("--dept",    default="",  help="Department")
    parser.add_argument("--desig",   default="",  help="Designation")
    parser.add_argument("--category", default="Teaching", choices=["Teaching", "Non-Teaching"],
                         help="Staff category (default: Teaching)")
    parser.add_argument("--hlc",     action="store_true", help="Also register as HLC team member")
    parser.add_argument("--images",  nargs="+",   help="Image file paths")
    parser.add_argument("--batch",   help="Batch folder path (see docstring for structure)")
    parser.add_argument("--live",    action="store_true", help="Enroll from config.camera.video_path")
    parser.add_argument("--demo",    action="store_true", help="Add demo faculty")
    parser.add_argument("--summary", action="store_true", help="Print enrollment status")
    args = parser.parse_args()

    enroller = FaceEnroller()

    if args.demo:
        enroller.add_demo_faculty()
    elif args.summary:
        enroller.print_summary()
    elif args.batch:
        enroller.enroll_from_folder(args.batch, default_category=args.category,
                                     default_hlc=args.hlc)
    elif args.live and args.id and args.name:
        enroller.enroll_live(args.id, args.name, args.dept,
                              staff_category=args.category, is_hlc=args.hlc)
    elif args.id and args.name and args.images:
        enroller.enroll_faculty(
            args.id, args.name, department=args.dept, designation=args.desig,
            staff_category=args.category, is_hlc=args.hlc, images=args.images
        )
    else:
        parser.print_help()
