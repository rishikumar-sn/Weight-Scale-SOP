"""Compare exported ONNX probabilities and preprocessing to a supplied checkpoint."""
from pathlib import Path
import argparse
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--onnx', type=Path, required=True)
    parser.add_argument('--previous', type=Path)
    parser.add_argument('--samples-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from backend.app.analysis.thread_classifier import JewelThreadClassifier

    torch.set_num_threads(2)
    model = torch.load(args.source, map_location='cpu', weights_only=False)['model'].float().eval()
    model.model[-1].export = True
    gate = JewelThreadClassifier(args.onnx)
    previous = JewelThreadClassifier(args.previous) if args.previous else None
    samples = sorted(p for p in args.samples_dir.rglob('*') if p.suffix.lower() in {'.png', '.jpg', '.jpeg'})
    assert samples, 'No verification images found'
    records = []
    for path in samples:
        image = cv2.imread(str(path))
        assert image is not None, path
        prepared = gate.prepare(image)
        reference_input = model.transforms(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))).numpy()[None]
        np.testing.assert_array_equal(prepared, reference_input)
        with torch.inference_mode():
            reference = model(torch.from_numpy(reference_input)).numpy().reshape(-1)
        start = time.perf_counter()
        result = gate.classify(image)
        seconds = time.perf_counter() - start
        actual = np.array([result['scores'][gate.names[i]] for i in range(len(gate.names))])
        np.testing.assert_allclose(actual, reference, atol=2e-4, rtol=2e-4)
        assert result['label'] == gate.names[int(np.argmax(reference))]
        normalized = path.parent.name.lower().replace('-', '_').replace(' ', '_')
        normalized = 'empty_testbed' if normalized == 'empty_test_bed' else normalized
        expected = normalized if normalized in set(gate.names.values()) else None
        row = dict(source=str(path), expected_label=expected, prediction=result,
                   pt_max_absolute_error=float(np.abs(actual-reference).max()), cpu_seconds=round(seconds, 4))
        if expected:
            row['correct_class'] = expected == result['label']
            row['correct_allow_block'] = result['passed'] == (expected in {'jewel_with_thread', 'jewel_without_thread'})
        if previous:
            row['previous_prediction'] = previous.classify(image)
        records.append(row)
        print(path.name, expected, '->', result['label'], round(result['confidence'], 4), flush=True)
    payload = dict(status='passed', samples=len(records), preprocessing_exact_match=True,
                   max_pt_onnx_error=max(r['pt_max_absolute_error'] for r in records),
                   correct_classes=sum(r.get('correct_class', False) for r in records),
                   correct_allow_block=sum(r.get('correct_allow_block', False) for r in records),
                   labelled_samples=sum(r['expected_label'] is not None for r in records), records=records,
                   note='Curated regression samples are not an unbiased accuracy benchmark.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in payload.items() if k!='records'}, indent=2))


if __name__ == '__main__':
    main()
