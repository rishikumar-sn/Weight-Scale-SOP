from PIL import Image

from backend.app.services.compression_service import ArtifactCompressionService


class RepositoryStub:
    def __init__(self, root):
        self.root = root

    def session_dir(self, capture_id):
        return self.root / capture_id


def test_completed_capture_is_compressed_without_pixel_changes(tmp_path):
    session = tmp_path / "capture-1" / "capture"
    session.mkdir(parents=True)
    source_path = session / "original.png"
    source = Image.new("RGB", (256, 256), "white")
    source.save(source_path, format="PNG", compress_level=0)
    expected_pixels = source.tobytes()
    before = source_path.stat().st_size

    service = ArtifactCompressionService(RepositoryStub(tmp_path))
    storage = service.finalize({"id": "capture-1"})

    with Image.open(source_path) as compressed:
        assert compressed.tobytes() == expected_pixels
    assert source_path.stat().st_size < before
    assert storage["lossless"] is True
    assert storage["processed_after_results"] is True
    assert storage["processed_after_analysis"] is True
    assert storage["saved_bytes"] > 0
    assert storage["presentation_artifacts"] == "compressed_on_demand"
    assert not (tmp_path / "capture-1/results/final").exists()


def test_jpeg_is_not_reencoded(tmp_path):
    session = tmp_path / "capture-2" / "capture"
    session.mkdir(parents=True)
    jpeg_path = session / "evidence.jpg"
    Image.new("RGB", (32, 32), "red").save(jpeg_path, format="JPEG")
    original_bytes = jpeg_path.read_bytes()

    ArtifactCompressionService(RepositoryStub(tmp_path)).finalize(
        {"id": "capture-2"}
    )

    assert jpeg_path.read_bytes() == original_bytes
