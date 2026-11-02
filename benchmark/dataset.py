"""Read local VinText without copying or modifying source images."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon


def read_image(path):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode {path}")
    return image


def read_annotations(path):
    objects = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split(",", 8)
        if len(parts) != 9:
            raise ValueError(f"{path}:{number}: expected 8 coordinates and transcription")
        values = [float(v) for v in parts[:8]]
        points = [values[i:i + 2] for i in range(0, 8, 2)]
        polygon = Polygon(points)
        if not polygon.is_valid or polygon.area <= 0 or not np.isfinite(values).all():
            raise ValueError(f"{path}:{number}: invalid polygon")
        text = parts[8].strip()
        objects.append({"polygon": points, "text": text, "ignore": text in {"###", "*"}})
    return objects


def prepare_dataset(root, manifest):
    root, manifest = Path(root), Path(manifest)
    images = sorted((root / "images").glob("*"))
    images = [p for p in images if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if not images:
        raise ValueError(f"No images in {root / 'images'}")
    rows, digest, used_labels = [], hashlib.sha256(), set()
    for path in images:
        label = root / "annotations" / f"gt_{path.stem.removeprefix('im')}.txt"
        annotations = read_annotations(label)
        used_labels.add(label.name)
        image = read_image(path)
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
        digest.update(label.read_bytes())
        rows.append({"id": path.stem, "image": path.resolve().as_posix(),
                     "width": image.shape[1], "height": image.shape[0], "annotations": annotations})
    extra = {p.name for p in (root / "annotations").glob("*.txt")} - used_labels
    if extra:
        raise ValueError(f"Orphan annotation files: {sorted(extra)[:10]}")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    audit = {"images": len(rows), "polygons": sum(len(r["annotations"]) for r in rows),
             "ignored": sum(a["ignore"] for r in rows for a in r["annotations"]),
             "sha256": digest.hexdigest(), "split": "test", "root": str(root.resolve())}
    manifest.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def load_manifest(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
