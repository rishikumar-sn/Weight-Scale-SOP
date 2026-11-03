"""Bundled CPU REAL/PRINT gate; original camera pixels, never masked cutouts."""
from __future__ import annotations
import hashlib
import json
import time
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

PRINT_MESSAGE = "Printed jewellery detected. Remove the printout, place physical jewellery on the testbed, and recapture."
PRINT_REVIEW_MESSAGE = "Jewellery authenticity could not be verified confidently. Check the items and recapture."
PRINT_UNAVAILABLE_MESSAGE = "Real/print verification is unavailable. Restore verification and recapture."


def blocks_real_print(state: dict) -> bool:
    check=state.get('real_print_check') or {}
    return (check.get('passed') is False
        or any(item.get('label') in {'PRINT','REVIEW'} or item.get('passed') is False for item in check.get('items',[]))
        or state.get('rejection_kind') in {'print_detected','real_print_review','real_print_unavailable'})


def padded_box(box,shape,padding):
    h,w=shape[:2];x,y,bw,bh=box
    x1,y1=round(x-bw*padding),round(y-bh*padding)
    x2,y2=x1+round(bw*(1+2*padding)),y1+round(bh*(1+2*padding))
    x1,y1=max(0,x1),max(0,y1);x2,y2=min(w,x2),min(h,y2)
    if x2<=x1 or y2<=y1: raise ValueError('Candidate does not intersect the original camera image')
    return [x1,y1,x2-x1,y2-y1]


class RealPrintClassifier:
    def __init__(self,model_path: Path):
        self.model_path=Path(model_path)
        self.config=json.loads(self.model_path.with_suffix('.json').read_text(encoding='utf-8'))
        c=self.config
        if c.get('format')!='real-print-v1' or c.get('labels')!=['REAL','PRINT'] or c.get('batch_order')!=['tight','context']:
            raise ValueError('Incompatible REAL/PRINT model metadata')
        if hashlib.sha256(self.model_path.read_bytes()).hexdigest()!=c.get('sha256'):
            raise ValueError('REAL/PRINT model checksum mismatch')
        self.size=int(c['image_size']);self.weight=float(c['fusion_weight'])
        self.threshold=float(c['threshold']);self.band=float(c['uncertainty_half_band'])
        if not (32<=self.size<=1024 and 0<=self.weight<=1 and 0<self.threshold<1 and 0<=self.band<min(self.threshold,1-self.threshold)):
            raise ValueError('Invalid REAL/PRINT inference configuration')
        if not (0<=float(c['tight_padding'])<=.6 and .15<=float(c['context_padding'])<=.6):
            raise ValueError('Invalid REAL/PRINT crop padding')
        self.mean=np.asarray(c['mean'],np.float32).reshape(3,1,1)
        self.std=np.asarray(c['std'],np.float32).reshape(3,1,1)
        if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or np.any(self.std<=0):
            raise ValueError('Invalid REAL/PRINT normalization')
        options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(self.model_path),sess_options=options,providers=['CPUExecutionProvider'])
        model_input=self.session.get_inputs()[0]
        if model_input.shape!=[2,3,self.size,self.size] or self.session.get_outputs()[0].shape!=[2,2]:
            raise ValueError('Unexpected REAL/PRINT ONNX input/output dimensions')
        self.input_name=model_input.name

    def crops(self,original,box):
        if original is None or original.ndim!=3 or original.shape[2]!=3 or original.size==0:
            raise ValueError('Original full-resolution BGR camera image is required')
        views=[];boxes=[]
        for padding in (self.config['tight_padding'],self.config['context_padding']):
            x,y,w,h=padded_box(box,original.shape,padding)
            views.append(original[y:y+h,x:x+w].copy());boxes.append([x,y,w,h])
        return views,boxes

    def prepare(self,crop):
        rgb=Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB))
        scale=self.size/max(rgb.size)
        rgb=rgb.resize((max(1,round(rgb.width*scale)),max(1,round(rgb.height*scale))),Image.Resampling.BILINEAR)
        canvas=Image.new('RGB',(self.size,self.size),(240,240,240))
        canvas.paste(rgb,((self.size-rgb.width)//2,(self.size-rgb.height)//2))
        tensor=np.asarray(canvas,dtype=np.float32).transpose(2,0,1)/255.
        return (tensor-self.mean)/self.std

    def classify(self,views):
        if len(views)!=2: raise ValueError('A tight/context pair is required')
        batch=np.ascontiguousarray(np.stack([self.prepare(view) for view in views]),dtype=np.float32)
        start=time.perf_counter()
        logits=np.asarray(self.session.run(None,{self.input_name:batch})[0])
        latency=(time.perf_counter()-start)*1000
        if logits.shape!=(2,2) or not np.isfinite(logits).all(): raise RuntimeError('Invalid REAL/PRINT model output')
        probabilities=np.exp(logits-logits.max(axis=1,keepdims=True))
        probabilities/=probabilities.sum(axis=1,keepdims=True)
        tight,context=map(float,probabilities[:,1]);fused=self.weight*tight+(1-self.weight)*context
        label='REVIEW' if self.band>0 and abs(fused-self.threshold)<=self.band else ('PRINT' if fused>=self.threshold else 'REAL')
        return dict(label=label,passed=label=='REAL',model=self.model_path.name,model_sha256=self.config['sha256'],
            dataset_version=self.config.get('dataset_version'),tight_print=tight,context_print=context,fused_print=fused,
            tight_real=1-tight,context_real=1-context,fused_real=1-fused,fusion_weight=self.weight,
            threshold=self.threshold,uncertainty_half_band=self.band,latency_ms=latency,input_source='original_camera_dual_crop')
