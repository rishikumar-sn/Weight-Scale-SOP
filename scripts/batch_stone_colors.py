"""Re-run a sampled batch of saved captures into an isolated output directory."""
from pathlib import Path
import argparse
import json
import sys
import time
from collections import Counter
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from backend.app.services.analysis_service import AnalysisService

class OutputRepository:
    def __init__(self, output):
        self.output = output
    def session_dir(self, capture_id):
        return self.output / capture_id

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit',type=int,default=30,help='0 runs all available stone captures')
    parser.add_argument('--output',type=Path,default=ROOT/'experiments'/'stone_color_batch')
    args = parser.parse_args()
    paths = sorted((ROOT/'data'/'sessions').glob('*/results/stones/item_*/result.json'))
    if args.limit and len(paths)>args.limit:
        paths = [paths[i] for i in np.linspace(0,len(paths)-1,args.limit,dtype=int)]
    args.output.mkdir(parents=True,exist_ok=True)
    service = AnalysisService(OutputRepository(args.output))
    records, errors, colors = [], [], Counter()
    try:
        for i,path in enumerate(paths,1):
            session = path.parents[3]
            try:
                state = json.loads((session/'manifest.json').read_text(encoding='utf-8'))
                index = int(path.parent.name.split('_')[-1])
                item = next(it for it in state['classification']['items'] if int(it['index'])==index)
                for mapping in [state['paths'],item['paths']]:
                    for key,value in list(mapping.items()):
                        local = session / ('items' if mapping is item['paths'] else 'capture') / Path(value).name
                        if local.exists():
                            mapping[key] = str(local.resolve())
                # Exercise both output modes on the same detector/color implementation.
                enabled = i % 2 == 0
                state.setdefault('settings_snapshot',{})['stone_analysis'] = {'enable_weight_estimation':enabled}
                start = time.perf_counter()
                result = service._run_stones(state,item)
                stored = json.loads(path.read_text(encoding='utf-8'))
                generated = json.loads(Path(result['report']).read_text(encoding='utf-8'))
                predictions = result.get('regions',[])
                colors.update(e['color'] for e in predictions)
                records.append(dict(session=session.name,item=index,enabled=enabled,seconds=round(time.perf_counter()-start,3),
                    stored_region_count=stored.get('stone_instance_count'),current_region_count=generated.get('stone_instance_count'),
                    found=result['found'],summary=result['summary'],colors=dict(Counter(e['color'] for e in predictions)),
                    sizes=dict(Counter(e['size'] for e in predictions)),image=result['image'],
                    weight_fields_present='estimated_weight_g' in result,
                    predictions=predictions))
            except Exception as exc:
                errors.append(dict(session=session.name,item=path.parent.name,error=str(exc)))
            print(f'{i}/{len(paths)} captures; {len(errors)} errors',flush=True)
    finally:
        service.shutdown()
    payload = dict(capture_count=len(records),error_count=len(errors),colors=dict(colors),
                   region_count_changes=sum(r['stored_region_count']!=r['current_region_count'] for r in records),
                   median_seconds=float(np.median([r['seconds'] for r in records])) if records else None,
                   records=records,errors=errors)
    (args.output/'results.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in payload.items() if k not in {'records','errors'}},indent=2))
    if errors:
        print(json.dumps(errors,indent=2))
        raise SystemExit(1)

if __name__=='__main__':
    main()
