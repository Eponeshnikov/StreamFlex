"""One fixed linear filter bank + one pilot-frozen common quantizer.

No ACF thinning, log, running RMS, conditional CDF, re-ranking, check-bit
rejection or rebalance. Exploratory raw-bit decorrelation/BDR study, not an
entropy/security estimate. All model selection uses the excluded pilot only.
"""
from __future__ import annotations

import argparse
import gc
import json
import pickle
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.fft import dct
from scipy.stats import norm, rankdata

from pages.key_generation.compute.nist.runner import run_all

OUT=Path('output_data/notes/keygen_linear_scheme_20261105')
PARENT=Path('output_data/notes/keygen_crypto_fix_20261101/parent_manifest.json')
BASE=Path('output_data/notes/keygen_streamlit_growing_20261102')
L=8
TARGET_BDR=.02
MIN_PILOT_BLOCKS=64


def read(path):
    with Path(path).open('rb') as f:return pickle.load(f)


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False,default=lambda a:a.item())+'\n')


def corr(a,b):
    if len(a)<30 or np.ptp(a)==0 or np.ptp(b)==0:return None
    return float(np.corrcoef(a,b)[0,1])


def acf(a,lags=128):
    a=np.asarray(a,float)
    return [corr(a[:-k],a[k:]) if len(a)>k else None for k in range(1,lags+1)]


def quantize(a,edges):
    return np.searchsorted(edges,a,side='right').astype(np.int64)


def ber(a,b,bits):
    if not len(a):return None
    ga=a^(a>>1);gb=b^(b>>1)
    return float(np.bitwise_count(ga^gb).sum()/(bits*len(a)))


def transform(blocks,kind):
    y=dct(blocks,type=2,norm='ortho',axis=1)
    if kind=='dct_dc_difference':
        y[1:,0]-=y[:-1,0].copy()
        return y[1:]
    return y


def blocks(raw,kind):
    records=[]
    for j in range(raw.status.shape[1]):
        ok=np.isfinite(raw.x['alice'][:,j])&np.isfinite(raw.x['bob'][:,j])
        ix=np.flatnonzero(ok);n=len(ix)//L
        if n<2:continue
        source=ix[:n*L].reshape(n,L)
        x={p:raw.dtau[p][source,j]*1e9 for p in ('alice','bob')}
        y={p:transform(a,kind) for p,a in x.items()}
        complete=source[1:,-1] if kind=='dct_dc_difference' else source[:,-1]
        records.append((j,complete,y))
    return records


def sequence(records,pivot,mode):
    events=[];models=[];tail=0
    for j,complete,y in records:
        train=complete<=pivot;use=complete>pivot
        if train.sum()<MIN_PILOT_BLOCKS:continue
        for k in range(L):
            a,b=y['alice'][train,k],y['bob'][train,k]
            common=np.r_[a,b]
            if np.ptp(common)==0:continue
            edges={m:np.quantile(common,np.arange(1,1<<m)/(1<<m)) for m in (1,2)}
            pilot_errors={m:ber(quantize(a,edges[m]),quantize(b,edges[m]),m) for m in (1,2)}
            if mode=='adaptive':
                eligible=[m for m in (1,2) if pilot_errors[m]<=TARGET_BDR]
                m=max(eligible,default=0)
            else:m=int(mode)
            rho=corr(a,b)
            gaussian_p1=float(np.arccos(np.clip(rho,-1,1))/np.pi) if rho is not None else None
            models.append(dict(profile=j,coefficient=k,pilot_blocks=int(train.sum()),bits=m,
                               pilot_BDR=pilot_errors,AB_correlation=rho,gaussian_sign_BDR=gaussian_p1,
                               model_note='sign formula exact only for centred jointly Gaussian input',
                               edges=edges[m].tolist() if m else []))
            if not m:continue
            aa=quantize(y['alice'][use,k],edges[m]);bb=quantize(y['bob'][use,k],edges[m])
            ga=aa^(aa>>1);gb=bb^(bb>>1)
            for step,ca,cb in zip(complete[use],ga,gb,strict=True):
                for bit in range(m-1,-1,-1):
                    events.append((int(step),j,k,m-1-bit,int((ca>>bit)&1),int((cb>>bit)&1)))
    events.sort(key=lambda row:row[:4])
    data=np.asarray(events,dtype=np.int64).reshape(-1,6)
    return data,models


def evaluate(data,models):
    a,b=data[:,4].astype(np.uint8),data[:,5].astype(np.uint8)
    n=len(a);parties={}
    for party,bits in [('alice',a),('bob',b)]:
        ac=acf(bits)
        planes=[]
        for j,k,bit in sorted({tuple(row) for row in data[:,[1,2,3]]}):
            chosen=(data[:,1]==j)&(data[:,2]==k)&(data[:,3]==bit)
            seq=bits[chosen]
            if len(seq)<200:continue
            values=acf(seq,16)
            planes.append(dict(profile=int(j),coefficient=int(k),bit=int(bit),n=len(seq),acf=values,
                               max_abs=max((abs(v) for v in values if v is not None),default=None)))
        prefix=bits[:65536]
        tests=[asdict(t) for t in run_all(prefix)] if len(prefix)>=100 else []
        parties[party]=dict(ones=float(bits.mean()) if n else None,acf=ac,
                            max_abs_ACF=max((abs(v) for v in ac if v is not None),default=None),
                            per_plane=planes,nist_bits=len(prefix),nist=tests,
                            first256=''.join(map(str,bits[:256].tolist())))
    return dict(bits=n,BDR=float(np.mean(a!=b)) if n else None,
                pilot_models=models,party=parties)


def worker(case):
    parent=json.loads(PARENT.read_text());raw=read(parent['caches'][case]['decor']['path'])
    raw.acf_pooled=raw.acf_profiles=raw.cca_curves=None
    baseline=read(BASE/f'{case}_calib.pkl');pivot=baseline.decision['step'] if baseline.decision else -1
    print('LOAD',case,raw.status.shape,pivot,flush=True)
    results=[]
    for kind in ['dct','dct_dc_difference']:
        rec=blocks(raw,kind)
        for mode in ['1','2','adaptive']:
            data,models=sequence(rec,pivot,mode)
            result=evaluate(data,models);result.update(case=case,transform=kind,mode=mode,pivot=pivot)
            np.savez_compressed(OUT/f'{case}_{kind}_{mode}.npz',alice=data[:,4].astype(np.uint8),
                                bob=data[:,5].astype(np.uint8),provenance=data[:,:4])
            results.append(result)
            write(OUT/f'{case}_progress.json',results)
            a=result['party']['alice'];fail=sum(t['passed'] is False for t in a['nist'])
            print('RESULT',case,kind,mode,result['bits'],result['BDR'],a['max_abs_ACF'],'NISTfail',fail,flush=True)
        del rec
    write(OUT/f'{case}.json',dict(case=case,baseline=dict(bits=sum(baseline.widths[baseline.kept]),BDR=None),
                                results=results))
    del raw;gc.collect()


def main():
    p=argparse.ArgumentParser();p.add_argument('--case');args=p.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.case:worker(args.case);return
    write(OUT/'design.json',dict(block=8,source='raw positive delay, receiver validity preserved',
        transforms=['orthonormal DCT8','DCT8 with DC first difference across adjacent blocks'],
        quantizer='one common pilot-frozen quantile threshold table per profile/coefficient',
        pilot='all blocks completed by the existing excluded production pilot boundary',
        modes=['1 bit','2 bits','max of 1/2 bits with pilot BDR <=2%; otherwise 0'],
        runtime_steps=['one fixed linear filter bank','one threshold quantizer and Gray mapping'],
        no=['ACF thinning','log','running RMS','conditional ranks','second CDF','check-bit rejection','rebalance'],
        criteria=['bit ACF lags1..128','plane ACF lags1..16 for n>=200','NIST first65536 bits','BDR','raw bit count'],
        limitations=['pilot parameters are public; no secrecy proof or final key length claimed',
                     'same records used previously; no independent transfer claim',
                     'no guarantee of Gaussianity from DCT; actual bits tested']))
    for case in ['kazan_v4_00','kazan_long_00','july_kaiser9_02','july_kaiser9_03']:
        subprocess.run([sys.executable,__file__,'--case',case],check=True)


if __name__=='__main__':main()
