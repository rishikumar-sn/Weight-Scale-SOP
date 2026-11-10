"""Compare anomaly exports on a reproducible, stratified saved-capture sample.

Saved decisions are reference evidence, not independently labelled ground truth.
No captures, database entries or model settings are modified.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True,
                        help='External baseline export; retired weights are not bundled in this project')
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--per-group', type=int, default=4)
    args = parser.parse_args()
    import cv2
    from backend.app.analysis.foreign_objects import ForeignObjectDetector
    groups = defaultdict(list)
    for manifest in sorted((ROOT/'data/sessions').glob('*/manifest.json')):
        state = json.loads(manifest.read_text(encoding='utf-8'))
        if not (manifest.parent/'capture/original.png').is_file() or not (manifest.parent/'capture/testbed_source_mask.png').is_file():
            continue
        labels = ','.join(sorted(set((state.get('classification') or {}).get('labels') or [])))
        anomalies = ','.join(sorted({d['label'] for d in (state.get('foreign_object_check') or {}).get('detections', [])}))
        group = state.get('rejection_kind') or anomalies or labels or state.get('status', 'unknown')
        groups[group].append((manifest, state))
    selected = []
    for group, entries in sorted(groups.items()):
        # Spread samples across the history instead of taking adjacent captures.
        indices = sorted({round(i*(len(entries)-1)/max(1, min(args.per_group,len(entries))-1))
                          for i in range(min(args.per_group, len(entries)))})
        selected.extend((group, *entries[i]) for i in indices)
    models = {'baseline': ForeignObjectDetector(args.baseline), 'candidate': ForeignObjectDetector(args.candidate)}
    report = {'limits': 'Saved captures and saved testbed masks; historical decisions are not independent ground truth. No labelled coin/bar validation set or live hardware test.',
              'models': {name: {'path': str(model.model_path), 'sha256': hashlib.sha256(model.model_path.read_bytes()).hexdigest(), 'classes': model.names,
                                'input': [model.input_height,model.input_width]} for name,model in models.items()}, 'rows': []}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    for group, manifest, state in selected:
        raw = cv2.imread(str(manifest.parent/'capture/original.png'))
        mask = cv2.imread(str(manifest.parent/'capture/testbed_source_mask.png'),0)
        if raw is None or mask is None or mask.shape != raw.shape[:2]:
            continue
        row = {'id':manifest.parent.name,'group':group,'saved_status':state.get('status'),
               'saved_check':state.get('foreign_object_check',{})}
        for name,model in models.items():
            start=time.perf_counter()
            check=model.detect(raw,mask,components=[])
            row[name]={'elapsed_ms':round((time.perf_counter()-start)*1000,1),**check}
        report['rows'].append(row)
        print(row['id'],group,[(name,row[name]['passed'],[d['label'] for d in row[name]['detections']],
                              [d['label'] for d in row[name]['recognized_objects']]) for name in models],flush=True)
        args.report.write_text(json.dumps(report,indent=2),encoding='utf-8')
    report['summary']={'captures':len(report['rows']), 'groups':dict(Counter(r['group'] for r in report['rows'])),
                       'gate_disagreements':[r['id'] for r in report['rows'] if r['baseline']['passed']!=r['candidate']['passed']]}
    args.report.write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
