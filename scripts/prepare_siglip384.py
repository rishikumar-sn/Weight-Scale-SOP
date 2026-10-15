"""Install the tested local export and migrate available labeled gallery images."""
from __future__ import annotations
import argparse,hashlib,json,shutil,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from Classification.jewelry_classifier import PROMPT_EXPANSION_VERSION, _normalize_vector
from Classification.siglip384_classifier import (
    ASSET_DIR,GALLERY_PATH,EMBEDDING_SPACE,MODEL_ID,ModelCorrectionGallery,Siglip384Classifier,
)
from scripts.calibrate_jewelry_classifier import discover_dataset

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--experiment',type=Path,default=ROOT/'experiments/siglip2_so400m_benchmark')
    parser.add_argument('--dataset',type=Path,default=Path('C:/Users/karth/Datasets/UniversalDatasetCollector'))
    args=parser.parse_args();experiment=args.experiment
    output=ROOT/'data/classification-evaluation/base384_migration';output.mkdir(parents=True,exist_ok=True)
    legacy=ROOT/'Classification/jewelry_correction_gallery.npz'
    legacy_hash=digest(legacy)
    if not (output/'patch32_gallery_backup.npz').exists():shutil.copy2(legacy,output/'patch32_gallery_backup.npz')
    ASSET_DIR.mkdir(parents=True,exist_ok=True)
    source=experiment/'base384_assets/model'
    integrity=json.loads((experiment/'base384_assets/model_integrity.json').read_text())
    for path in source.iterdir():
        if not path.is_file():continue
        target=ASSET_DIR/path.name
        expected=integrity.get(path.name)
        if not target.exists() or target.stat().st_size!=path.stat().st_size or (expected and digest(target)!=expected):
            shutil.copy2(path,target)
        if expected and digest(target)!=expected:raise RuntimeError('Model integrity failed: '+str(target))
    print('Local base384 vision, text and tokenizer assets installed and verified.',flush=True)
    for name in ['jewelry_prompts.json','foreign_object_prompts.json']:
        if (ROOT/'Classification'/name).read_bytes()!=(experiment/'base384'/name).read_bytes():
            raise RuntimeError('Prompts changed since benchmark; rebuild text caches before migrating.')
    obj=Siglip384Classifier.__new__(Siglip384Classifier)
    obj.prompt_path=ROOT/'Classification/jewelry_prompts.json'
    obj.group_prompt_config,obj.class_prompt_definitions,obj.prompt_config=obj._load_prompt_config()
    prompt_hash=obj._build_prompt_hash()
    cache=np.load(experiment/'base384/text_vectors.npz')
    np.savez(ASSET_DIR/'jewelry_text_embeddings_cache.npz',
             **{key:cache[key] for key in ['labels','embeddings','class_groups','group_labels','group_embeddings']},
             text_model_id=np.array([MODEL_ID]),prompt_hash=np.array([prompt_hash]))
    foreign=json.loads((ROOT/'Classification/foreign_object_prompts.json').read_text())
    payload=json.dumps({'config':foreign,'model':MODEL_ID,'expansion':PROMPT_EXPANSION_VERSION,
                        'object_names':list(foreign['objects'])},sort_keys=True)
    np.savez_compressed(ASSET_DIR/'foreign_object_text_embeddings_cache.npz',
                        embeddings=cache['foreign'],prompt_hash=np.array(hashlib.sha256(payload.encode()).hexdigest()))
    gallery=ModelCorrectionGallery(GALLERY_PATH)
    inventory=json.loads((experiment/'inventory.json').read_text())['rows']
    historical=0
    for row in inventory:
        if not row['truth']:continue
        path=experiment/'base384/embeddings'/(hashlib.sha256(row['id'].encode()).hexdigest()+'.npz')
        with np.load(path) as entry:
            if str(entry['image_sha256'])!=row['image_sha256']:raise RuntimeError('Stale history embedding')
            gallery.add(entry['merged'],row['truth'],save=False)
        historical+=1
    gallery.save()
    print(f'Seeded {historical} confirmed history images; {len(gallery.labels)} unique gallery entries.',flush=True)
    classifier=Siglip384Classifier()
    # Add to the same gallery object used below, retaining corrections on reruns.
    classifier.gallery=gallery
    records,rejected=discover_dataset(args.dataset,70,'weight-integration-v1')
    calibration=[record for record in records if record.split=='calibration']
    if not calibration:raise RuntimeError('No existing calibration dataset found: '+str(args.dataset))
    paths=[(p,r.expected_label) for r in calibration for p in [r.roi_path,r.full_path] if p]
    cache_dir=output/'embeddings';cache_dir.mkdir(exist_ok=True)
    for index,(name,label) in enumerate(paths,1):
        path=Path(name);key=hashlib.sha256((str(path)+':'+digest(path)).encode()).hexdigest()
        target=cache_dir/(key+'.npz')
        try:
            with np.load(target) as entry:
                if str(entry['embedding_space'])!=EMBEDDING_SPACE:raise ValueError('Stale cache')
                vector=entry['embedding']
                if vector.shape!=(768,) or not np.isfinite(vector).all():raise ValueError('Invalid vector')
        except (OSError,ValueError,KeyError):
            vector=classifier.embedding_for_path(path)
            temporary=target.with_suffix('.pending.npz')
            np.savez(temporary,embedding=vector,embedding_space=np.array(EMBEDDING_SPACE))
            temporary.replace(target)
        gallery.add(vector,label,save=False)
        if index%10==0 or index==len(paths):
            gallery.save();print(f'Calibration migration {index}/{len(paths)}; gallery {len(gallery.labels)} entries',flush=True)
    gallery.save()
    assert digest(legacy)==legacy_hash,'Legacy gallery changed during migration'
    report={'model':MODEL_ID,'embedding_space':EMBEDDING_SPACE,'historical_confirmed_images':historical,
            'calibration_captures':len(calibration),'calibration_images':len(paths),'gallery_entries':len(gallery.labels),
            'label_counts':{label:gallery.labels.count(label) for label in sorted(set(gallery.labels))},
            'original_patch32_gallery_sha256':legacy_hash,'original_patch32_gallery_unchanged':True,
            'rejected_dataset_files':rejected}
    (output/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
