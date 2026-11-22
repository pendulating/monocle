#!/usr/bin/env python
"""F10.18: where do the words of the paper sit on the sense axis, and how often is a place probe correct? CPU only.

Part 1 reads the `cool` token at layers 20 and 24. It gives each paper word its rank on axis 1
(social end first) and its borough t, with axis 1 in and out. Part 2 gives the share of images
with a positive margin on 4 place probes, by borough. All 591 images are in New York City, thus
the `in New York` row is an accuracy measure.

Output: `research/figures/f10/axes/axes_claim_words.csv`, `research/figures/f10/probe/probe_place.csv`.

Use:
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f10/hop_claim_axis.py
"""
import sys,numpy as np,pandas as pd
sys.path[:0]=['research/scripts/f10','research/notebooks/cvpr']
from analyze_logit import load, shard_reads
from analyze_probe import margins
from hop_borough import BOROUGHS, two_sample_t
pd.set_option('display.width',220,'display.max_rows',200,'display.max_colwidth',90,'display.max_columns',30)
out=[]
for model in ['gemma-4-31b','qwen3.5-9b']:
    d=load(model); im=d['images']; head=d['head']; tok=d['store'].manifest['tail_tokens']
    w=d['store'].vocab['token'].str.strip().str.lower().to_numpy(); at={x:i for i,x in enumerate(w)}
    a=tok.index(' cool'); 
    codes=pd.factorize(im['nta2020'])[0]; bor=im.groupby(codes)['borough'].first().to_numpy()
    fs=shard_reads(d)
    for layer in (20,24):
        b=head['layers'].index(layer)
        x=np.concatenate([np.array(np.load(f,mmap_mode='r')[:,a,b]) for f in fs]).astype(np.float32)
        x=(x-x.mean(-1,keepdims=True))/x.std(-1,keepdims=True)
        m=np.stack([x[codes==g].mean(0) for g in range(codes.max()+1)])
        c=m-m.mean(0,keepdims=True); e,u=np.linalg.eigh(c@c.T); u1=u[:,-1]; s=np.sqrt(e[-1]); v=(u1@c)/s
        # sign: social positive
        soc=[at[k] for k in ('hipster','boutique','bohemian') if k in at]
        if v[soc].mean()<0: v=-v; u1=-u1
        rank=(-v).argsort().argsort()+1
        res=c-np.outer(u1*s,v)
        for bn in BOROUGHS:
            t0=two_sample_t(c,bor==bn); t1=two_sample_t(res,bor==bn)
            rt0=(-t0).argsort().argsort()+1; rt1=(-t1).argsort().argsort()+1
            for wd in "graffiti fucking shopping gallery homeless porn anarch illegal vandalism gangs tattoo bodega slums boutiques prostit slum market traff marijuana prostitutes bistro boutique instagram social nightclubs seawater scientist hydrological equatorial humidity".split():
                if wd in at:
                    i=at[wd]; out.append(dict(model=model,layer=layer,borough=bn,word=wd,axis_rank=rank[i],axis_load=round(float(v[i]),4),t=round(float(t0[i]),2),t_rank=rt0[i],t_after=round(float(t1[i]),2),t_rank_after=rt1[i]))
o=pd.DataFrame(out); o.to_csv('research/figures/f10/axes/axes_claim_words.csv',index=False)
g=o[(o.model=='gemma-4-31b')]
for bn,wds in [('Bronx','graffiti fucking shopping gallery homeless porn anarch illegal vandalism gangs tattoo'),('Brooklyn','fucking boutique instagram bodega social nightclubs slums boutiques prostit slum market traff marijuana prostitutes bistro'),('Staten Island','seawater scientist hydrological'),('Queens','equatorial humidity')]:
    print(g[(g.borough==bn)&g.word.isin(wds.split())].sort_values(['layer','axis_rank']).drop(columns=['model']).to_string(index=False))
print('axis rank of all claim words, Gemma L24: top 1000 share', (g[(g.layer==24)&(g.borough=='Bronx')].axis_rank<=1000).mean())
rows=[]
for model in ['gemma-4-31b','qwen3.5-9b']:
    p=margins(model); d=load(model); im=d['images'].set_index('image_idx'); b=im['borough'].reindex(p.index)
    for w in ['in New York','in a city','urban','outdoors']:
        rows.append({'model':model,'word':w,'all':(p[w]>0).mean()}|((p[w]>0).groupby(b).mean()).to_dict())
pd.DataFrame(rows).to_csv('research/figures/f10/probe/probe_place.csv',index=False)
print(pd.DataFrame(rows).round(3).to_string(index=False))
