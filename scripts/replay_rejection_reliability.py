"""Replay the reported captures in temporary storage and repair review overlays.

Historical model decisions stay unchanged. --repair-reviews adds region metadata
and replaces only rejected review presentations, keeping backups in the audit.
"""
from pathlib import Path
from copy import deepcopy
import argparse
import json
import shutil
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repair-reviews',action='store_true')
    args=parser.parse_args()
    import cv2
    from backend.app.analysis.vision import separate_jewellery_items
    from backend.app.analysis.rejection_evidence import source_bbox, rejection_regions, rejection_summary
    from backend.app.analysis.annotations import render_capture_review
    from backend.app.services.analysis_service import AnalysisService
    from backend.app.storage.repository import CaptureRepository
    ids=['20261110_125324_16877aa0','20261110_125422_a771ae9c','20261110_125453_2e1af416']
    out=ROOT/'data/audits/rejection_reliability_20261110';out.mkdir(parents=True,exist_ok=True)
    report=[]
    with tempfile.TemporaryDirectory(prefix='rejection-replay-') as temporary:
        repository=CaptureRepository(Path(temporary)/'app.db',Path(temporary)/'sessions')
        service=AnalysisService(repository)
        try:
            for sid,expected in zip(ids,['rejected','awaiting_confirmation','awaiting_confirmation']):
                manifest=ROOT/'data/sessions'/sid/'manifest.json'
                saved=json.loads(manifest.read_text(encoding='utf-8'))
                before_status=saved['status']
                paths=saved['paths']
                if saved['status']=='rejected':
                    settings=(((saved.get('settings_snapshot') or {}).get('analysis') or {}).get('item_separation') or {})
                    components=separate_jewellery_items(cv2.imread(paths['working']),cv2.imread(paths['mask'],0),
                        min_area_px=int(settings.get('min_area_px',350)),min_area_ratio=float(settings.get('min_area_ratio',.00045)),
                        padding_px=int(settings.get('padding_px',18)),valid_area_mask=cv2.imread(paths['testbed_mask'],0))
                    saved['foreground_components']=[{'source_component_index':i,'bbox':item['bbox'],
                        'source_bbox':source_bbox(saved,item['bbox'])} for i,item in enumerate(components,1)]
                    saved['rejection_regions']=rejection_regions(saved)
                    saved['rejection_reason']=rejection_summary(saved,saved['rejection_regions'])
                    corrected=render_capture_review(cv2.imread(paths['original']),saved,[],rejected=True)
                    cv2.imwrite(str(out/f'{sid}_recorded_decision_review.jpg'),corrected)
                    if args.repair_reviews:
                        backup=out/'before'/sid;backup.mkdir(parents=True,exist_ok=True)
                        if not (backup/'manifest.json').exists():
                            shutil.copy2(manifest,backup/'manifest.json')
                            shutil.copy2(paths['evidence'],backup/Path(paths['evidence']).name)
                        assert saved['status']==before_status
                        assert cv2.imwrite(paths['evidence'],corrected)
                        # Do not alter status, classification, recorded inference scores,
                        # captured time or database fields just to repair a presentation.
                        pending=manifest.with_suffix('.review.tmp')
                        pending.write_text(json.dumps(saved,indent=2),encoding='utf-8');pending.replace(manifest)
                replay=deepcopy(saved)
                for key in ['foreign_object_check','component_presence_check','thread_only_check','real_print_check',
                            'rim_edge_check','rejection_reason','rejection_kind','rejection_regions','placement_check']:
                    replay.pop(key,None)
                replay.update(status='captured',classification={},result={},route={})
                # Keep every original input immutable; all new evidence goes to temp.
                replay['paths']=deepcopy(paths)
                session=repository.session_dir(sid);(session/'capture').mkdir(parents=True,exist_ok=True)
                repository.save(replay)
                result=service.classify(sid)
                assert result['status']==expected,(sid,result.get('rejection_reason'))
                rendered=cv2.imread(result['paths']['evidence'])
                assert rendered is not None
                cv2.imwrite(str(out/f'{sid}_corrected_model_replay.jpg'),rendered)
                report.append(dict(capture_id=sid,recorded_status=before_status,new_model_replay_status=result['status'],
                    rejection_kind=result.get('rejection_kind'),rejection_regions=result.get('rejection_regions',[]),
                    recorded_regions=saved.get('rejection_regions',[]),labels=(result.get('classification') or {}).get('labels',[]),
                    print_checks=[{k:d.get(k) for k in ('item_index','label','fused_print','model_sha256')} for d in
                                  (result.get('real_print_check') or {}).get('items',[])]))
                print(sid,before_status,'->',result['status'],result.get('rejection_reason',''),flush=True)
        finally:service.shutdown()
    (out/'three_capture_replay.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()
