"""Cyclomedia data handler for OCR pipeline.

Handles the Cyclomedia street view imagery directory structure:
- Base path contains location group directories (e.g., W0CGT, W0CPF)
- Each group contains location directories (e.g., W0ETZ9YV)
- Each location contains: complete.ok, depthmaps_faces/, faces/, manifest.json
- faces/ contains 6 cube face images: B.jpg, D.jpg, F.jpg, L.jpg, R.jpg, U.jpg
"""

import os
import json
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd
from omegaconf import DictConfig

from .base import OCRDataHandler


class CyclomediaHandler(OCRDataHandler):
    """Data handler for Cyclomedia street view imagery.

    Extracts metadata:
    - location_group: Top-level directory (e.g., "W0ETZ")
    - location_id: Full location identifier (e.g., "W0ETZ9YV")
    - face: Cube face identifier (B, D, F, L, R, U)
    """

    # Cube face meanings for reference
    FACE_DESCRIPTIONS = {
        "B": "Back",
        "D": "Down",
        "F": "Front",
        "L": "Left",
        "R": "Right",
        "U": "Up",
    }

    def load_dataset(self, cfg: DictConfig) -> pd.DataFrame:
        """Load Cyclomedia image paths into a DataFrame.

        Expects cfg.data to contain:
        - image_path: Base path to Cyclomedia data (e.g., /share/ju/cyclomedia/raw/bronx_2025_1k)

        Returns:
            pd.DataFrame with columns: image_path, sample_id, location_group, location_id, face
        """
        data_cfg = getattr(cfg, "data", None)
        if data_cfg is None:
            raise ValueError("Configuration missing 'data' section")

        image_path = getattr(data_cfg, "image_path", None)
        if not image_path:
            raise ValueError("data.image_path must be specified for Cyclomedia handler")

        image_path = os.path.abspath(os.path.expanduser(str(image_path)))

        if not os.path.isdir(image_path):
            raise ValueError(f"Cyclomedia base path does not exist: {image_path}")

        print(f"[cyclomedia] Loading images from: {image_path}", flush=True)

        # Auto-detect directory level and collect all faces directories
        faces_dirs = self._discover_faces_dirs(image_path)

        if not faces_dirs:
            raise ValueError(f"No faces directories found under: {image_path}")

        print(f"[cyclomedia] Found {len(faces_dirs)} faces directories", flush=True)
        if len(faces_dirs) <= 10:
            for fd in faces_dirs:
                print(f"[cyclomedia]   - {fd}", flush=True)
        else:
            for fd in faces_dirs[:5]:
                print(f"[cyclomedia]   - {fd}", flush=True)
            print(f"[cyclomedia]   ... and {len(faces_dirs) - 5} more", flush=True)

        # Get face filter from config (default: all faces)
        faces_filter = getattr(data_cfg, "faces_filter", None)
        if faces_filter:
            if isinstance(faces_filter, str):
                faces_filter = [f.strip().upper() for f in faces_filter.split(",")]
            else:
                faces_filter = [f.upper() for f in faces_filter]
            print(f"[cyclomedia] Filtering to faces: {faces_filter}", flush=True)

        # Collect all image paths with metadata
        rows = []
        for faces_dir in faces_dirs:
            for fname in os.listdir(faces_dir):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                full_path = os.path.join(faces_dir, fname)
                if not os.path.isfile(full_path):
                    continue

                metadata = self.extract_metadata(full_path)
                face = metadata.get("face")

                # Apply face filter
                if faces_filter and (not face or face not in faces_filter):
                    continue

                location_id = metadata.get("location_id", "")
                sample_id = f"{location_id}_{face}" if location_id and face else os.path.basename(full_path)

                rows.append({
                    "image_path": full_path,
                    "sample_id": sample_id,
                    **metadata,
                })

        df = pd.DataFrame(rows)
        if df.empty:
            df = pd.DataFrame(columns=["image_path", "sample_id", "location_group", "location_id", "face"])

        print(
            json.dumps({
                "cyclomedia_handler": {
                    "event": "dataset_loaded",
                    "count": len(df),
                    "faces_dirs": len(faces_dirs),
                    "base_path": image_path,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            }),
            flush=True,
        )
        print(f"[cyclomedia] Dataset ready: {len(df)} images", flush=True)

        # Apply sample limit if configured
        sample_n = getattr(getattr(cfg, "runtime", None), "sample_n", None)
        if isinstance(sample_n, int) and sample_n > 0:
            df = df.head(sample_n)
            print(f"[cyclomedia] Applied sample limit: {sample_n}", flush=True)

        return df

    def _discover_faces_dirs(self, image_path: str) -> list:
        """Auto-detect directory level and collect all faces directories.

        Possible levels:
          1. Faces directory: .../faces/ -> use directly
          2. Single location: .../W0ETZ9YV/ -> use faces/
          3. Location group: .../W0ETZ/ -> collect all */faces/
          4. Full dataset: .../bronx_2025_1k/ -> collect all */*/faces/
        """
        base_name = os.path.basename(image_path)
        faces_dirs = []

        if base_name == "faces":
            faces_dirs = [image_path]
        elif os.path.isdir(os.path.join(image_path, "faces")):
            faces_dirs = [os.path.join(image_path, "faces")]
        else:
            subdirs = [d for d in os.listdir(image_path)
                       if os.path.isdir(os.path.join(image_path, d))]

            if subdirs:
                first_subdir = os.path.join(image_path, subdirs[0])
                if os.path.isdir(os.path.join(first_subdir, "faces")):
                    # Location group level
                    for subdir in subdirs:
                        faces_path = os.path.join(image_path, subdir, "faces")
                        if os.path.isdir(faces_path):
                            faces_dirs.append(faces_path)
                else:
                    # Full dataset level
                    for group_dir in subdirs:
                        group_path = os.path.join(image_path, group_dir)
                        if os.path.isdir(group_path):
                            loc_dirs = [d for d in os.listdir(group_path)
                                        if os.path.isdir(os.path.join(group_path, d))]
                            for loc_dir in loc_dirs:
                                faces_path = os.path.join(group_path, loc_dir, "faces")
                                if os.path.isdir(faces_path):
                                    faces_dirs.append(faces_path)

        return faces_dirs

    def extract_metadata(self, path: str) -> Dict[str, Any]:
        """Extract Cyclomedia-specific metadata from image path.

        Expected path format: .../base_path/{location_group}/{location_id}/faces/{face}.jpg

        Examples:
        - /share/ju/cyclomedia/raw/bronx_2025_1k/W0ETZ/W0ETZ9YV/faces/B.jpg
          -> location_group: W0ETZ, location_id: W0ETZ9YV, face: B
        """
        metadata = {
            "location_group": None,
            "location_id": None,
            "face": None,
        }

        if not path:
            return metadata

        try:
            path_parts = path.replace("\\", "/").split("/")

            if "faces" in path_parts:
                faces_idx = path_parts.index("faces")

                # Face is the filename without extension
                if faces_idx + 1 < len(path_parts):
                    face_file = path_parts[faces_idx + 1]
                    face = os.path.splitext(face_file)[0].upper()
                    if face in self.FACE_DESCRIPTIONS:
                        metadata["face"] = face

                # Location ID is the directory containing faces/
                if faces_idx >= 1:
                    metadata["location_id"] = path_parts[faces_idx - 1]

                # Location group is the directory above location_id
                if faces_idx >= 2:
                    metadata["location_group"] = path_parts[faces_idx - 2]

        except Exception:
            pass

        return metadata

    def get_manifest(self, location_path: str) -> Optional[Dict[str, Any]]:
        """Load manifest.json for a location if it exists.

        Args:
            location_path: Path to location directory (containing faces/)

        Returns:
            Manifest data or None if not found
        """
        manifest_path = os.path.join(location_path, "manifest.json")
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return None
