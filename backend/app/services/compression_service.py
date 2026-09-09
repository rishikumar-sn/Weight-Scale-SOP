from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from PIL import Image


class ArtifactCompressionService:
    """Losslessly compact the stored images belonging to a completed capture."""

    def __init__(self, repository) -> None:
        self.repository = repository

    @staticmethod
    def compress_png_losslessly(path: Path) -> dict[str, Any]:
        """Repack a PNG at maximum DEFLATE compression without changing pixels."""
        before = path.stat().st_size
        temporary = path.with_name(f"{path.name}.compressed.tmp")
        try:
            with Image.open(path) as source:
                source.load()
                original_mode = source.mode
                original_size = source.size
                original_pixels = source.tobytes()
                save_options: dict[str, Any] = {
                    "format": "PNG",
                    "optimize": True,
                    "compress_level": 9,
                }
                for key in ("icc_profile", "exif", "dpi", "transparency"):
                    if key in source.info:
                        save_options[key] = source.info[key]
                source.save(temporary, **save_options)

            with Image.open(temporary) as candidate:
                candidate.load()
                pixels_match = (
                    candidate.mode == original_mode
                    and candidate.size == original_size
                    and candidate.tobytes() == original_pixels
                )
            if not pixels_match:
                raise RuntimeError("PNG verification failed: decoded pixels changed")

            candidate_size = temporary.stat().st_size
            if candidate_size < before:
                os.replace(temporary, path)
                after = candidate_size
            else:
                temporary.unlink(missing_ok=True)
                after = before
            return {
                "path": str(path),
                "before_bytes": before,
                "after_bytes": after,
                "saved_bytes": before - after,
            }
        finally:
            temporary.unlink(missing_ok=True)

    def finalize(self, state: dict[str, Any]) -> dict[str, Any]:
        """Run only after inference, then create compact files used by the UI."""
        capture_id = str(state["id"])
        session_dir = self.repository.session_dir(capture_id).resolve()
        errors: list[dict[str, str]] = []
        files: list[dict[str, Any]] = []

        # JPEG is already compressed. Re-encoding it with Pillow would be lossy,
        # so only PNG containers are repacked after every model has finished.
        for path in sorted(session_dir.rglob("*.png")):
            try:
                files.append(self.compress_png_losslessly(path))
            except Exception as exc:  # A compression miss must not hide results.
                errors.append({"path": str(path), "message": str(exc)})

        before_bytes = sum(int(item["before_bytes"]) for item in files)
        after_bytes = sum(int(item["after_bytes"]) for item in files)
        return {
            "status": "complete" if not errors else "partial",
            "algorithm": "PNG DEFLATE level 9 (pixel-verified); PDF compressed streams",
            "lossless": True,
            "processed_after_results": True,
            "processed_after_analysis": state.get("capture_type") != "tare",
            "files_processed": len(files),
            "before_bytes": before_bytes,
            "after_bytes": after_bytes,
            "saved_bytes": before_bytes - after_bytes,
            # These large presentation artifacts stay out of persistent storage.
            # Their existing endpoints generate optimized streams on demand.
            "presentation_artifacts": "compressed_on_demand",
            "errors": errors,
        }
