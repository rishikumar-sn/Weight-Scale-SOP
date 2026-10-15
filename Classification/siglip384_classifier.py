"""Local FP32 base-patch16-384 encoder with the production decision pipeline."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import onnxruntime as ort
import torch
from transformers import SiglipImageProcessor, Siglip2Tokenizer

from Classification.jewelry_classifier import (
    CorrectionGallery, JewelryZeroShotClassifier, DEFAULT_PROMPT_FILE,
)

MODEL_ID = "google/siglip2-base-patch16-384"
MODEL_REVISION = "c066e994c50fc61679543dfbd12e6c3f0f5b6b47"
EMBEDDING_SPACE = f"{MODEL_ID}@{MODEL_REVISION}"
ASSET_DIR = Path(__file__).resolve().parent / "siglip2-base-patch16-384"
GALLERY_PATH = Path(__file__).resolve().parent / "jewelry_correction_gallery.base384.npz"


class ModelCorrectionGallery(CorrectionGallery):
    """Prevent accidental reuse of a different model's 768-dimensional vectors."""

    def load(self) -> None:
        if self.gallery_path.exists():
            with np.load(self.gallery_path, allow_pickle=False) as data:
                space = str(data["embedding_space"].item()) if "embedding_space" in data else ""
                if space != EMBEDDING_SPACE:
                    raise ValueError(f"Incompatible base384 gallery: {self.gallery_path}")
        super().load()

    def save(self) -> None:
        if self.embeddings is None:
            return
        self.gallery_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.gallery_path.with_name(self.gallery_path.stem + ".pending.npz")
        np.savez(temporary, embeddings=self.embeddings, labels=np.array(self.labels),
                 embedding_space=np.array(EMBEDDING_SPACE))
        temporary.replace(self.gallery_path)


class _Processor:
    def __init__(self, directory: Path) -> None:
        self.image = SiglipImageProcessor.from_pretrained(directory, local_files_only=True)
        self.tokenizer = Siglip2Tokenizer.from_pretrained(directory, local_files_only=True)

    def __call__(self, *, images=None, text=None, **kwargs):
        if images is not None:
            return self.image(images=images, **kwargs)
        kwargs.setdefault("max_length", 64)
        return self.tokenizer(text=text, **kwargs)


class _OnnxTextModel:
    def __init__(self, path: Path) -> None:
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        self.session = ort.InferenceSession(str(path), sess_options=options,
                                            providers=["CPUExecutionProvider"])

    def eval(self):
        return self

    def get_text_features(self, input_ids, **_kwargs) -> torch.Tensor:
        ids = input_ids.cpu().numpy().astype(np.int64)
        outputs = [self.session.run(None, {"input_ids": ids[start:start + 4]})[0]
                   for start in range(0, len(ids), 4)]
        return torch.from_numpy(np.concatenate(outputs).astype(np.float32))


class Siglip384Classifier(JewelryZeroShotClassifier):
    def __init__(self, *, providers: list[str] | None = None,
                 asset_dir: str | Path = ASSET_DIR,
                 gallery_path: str | Path = GALLERY_PATH) -> None:
        self.asset_dir = Path(asset_dir)
        super().__init__(
            onnx_model_path=self.asset_dir / "vision_model.onnx",
            prompt_path=Path(__file__).resolve().parent / DEFAULT_PROMPT_FILE,
            text_model_id=MODEL_ID,
            embedding_cache_path=self.asset_dir / "jewelry_text_embeddings_cache.npz",
            providers=providers,
            gallery_path=gallery_path,
            foreign_embedding_cache_path=self.asset_dir / "foreign_object_text_embeddings_cache.npz",
        )

    def _load_processor(self):
        return _Processor(self.asset_dir)

    def _load_text_model(self):
        return _OnnxTextModel(self.asset_dir / "text_model.onnx")

    def _make_gallery(self) -> CorrectionGallery:
        return ModelCorrectionGallery(self.gallery_path)
