"""Add background-count and historical Real-Scope audits after primary diagnostics."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import json
import torch
import numpy as np

ROOT=Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    c=read(ROOT/'diagnostics/check_c.json')
    background={}
    for dataset in ('Toys','Beauty','Sports'):
        source=Path(c[dataset]['source'])
        with np.load(source/'prepared/train.npz') as training:
            targets=training['target'].copy()
            history=training['seq'].copy()
        n=c[dataset]['item_count']
        bo=np.bincount(targets,minlength=n)
        valid=history[(history>=0)&(history<n)]
        histories=np.bincount(valid,minlength=n)
        with np.load(ROOT/'partitions'/dataset/'partitions.npz') as partitions:
            bn=partitions['bn_counts'].copy()
        bases={name:np.sqrt(counts+1.)/np.sqrt(counts+1.).sum() for name,counts in [('bo',bo),('bn',bn)]}
        background[dataset]={'bo_total':int(bo.sum()),'bn_total':int(bn.sum()),
            'bo_mean_count':float(bo.mean()),'bn_mean_count':float(bn.mean()),
            'bo_zero_items':int((bo==0).sum()),'bn_zero_items':int((bn==0).sum()),
            'bo_zero_but_history_at_least5':int(((bo==0)&(histories>=5)).sum()),
            'base_total_variation':float(.5*np.abs(bases['bo']-bases['bn']).sum()),
            'interpretation':'More observations and fewer zero counts do not establish ranking improvement or a causal benefit.'}
        c[dataset]['historical_real_scope']={'status':'No compatible saved partition found for this current Grow item mapping; historical Toys partition reported separately.'}
    historical={}
    old_root=Path('E:/Choice/runs/paper_newchoice_20261031')
    for dataset in ('Toys','Beauty','Sports'):
        candidates=sorted((old_root/'scopes'/dataset).glob('real_*.pt'),key=lambda p:int(p.stem.split('_')[-1]))
        if not candidates:
            historical[dataset]={'status':'not found'}
            continue
        old_path=candidates[0]
        artifact=torch.load(old_path,map_location='cpu',weights_only=True)
        old_base=artifact['base'].double().numpy()
        old_base/=old_base.sum()
        with np.load(old_root/'data'/dataset/'prepared/train.npz') as training:
            n=len(old_base)
            positions=np.concatenate((training['seq'].ravel(),training['target']))
            counts=np.bincount(positions[(positions>=0)&(positions<n)],minlength=n)
        boundary=int(np.sort(counts)[::-1][int(np.ceil(.2*n))-1])
        head=counts>=boundary
        levels=[]
        for level,labels in enumerate(artifact['blocks'].numpy()):
            sizes=np.bincount(labels)
            mass=np.bincount(labels,weights=old_base)
            ratio=np.bincount(labels,weights=head)/sizes
            levels.append({'level':level,'clusters':len(sizes),'size_min':int(sizes.min()),'size_max':int(sizes.max()),
                'block_mass_mean':float(mass.mean()),'block_mass_std':float(mass.std()),
                'block_mass_min':float(mass.min()),'block_mass_max':float(mass.max()),'head_fraction_std':float(ratio.std())})
        historical[dataset]={'source':str(old_path),'dataset':dataset,'levels':levels,
            'compatibility':'Prepared training sequences and targets differ from current Grow inputs. An original-ID correspondence has not been established; labels are not applied to current users or embeddings.',
            'background':'own historical artifact background; not the current bn background',
            'fidelity_and_current_validation_predictability':None}
        c[dataset]['historical_real_scope']={'status':'Historical partition exists; its own-mapping statistics are reported separately.',
            'source':str(old_path),'compatible_mapping_verified':False}
    (ROOT/'diagnostics/background_counts.json').write_text(json.dumps(background,indent=2)+'\n',encoding='utf-8')
    (ROOT/'diagnostics/historical_real_scope.json').write_text(json.dumps(historical,indent=2)+'\n',encoding='utf-8')
    (ROOT/'diagnostics/check_c.json').write_text(json.dumps(c,indent=2)+'\n',encoding='utf-8')
    # Regenerate primary tables using the corrected provenance before appending audits.
    from diagnostics_step0 import render_report,markdown_table
    render_report(read(ROOT/'diagnostics/check_a.json'),read(ROOT/'diagnostics/check_b.json'),c)
    text=(ROOT/'diagnostics/report.md').read_text(encoding='utf-8')
    text+='\n## Additional audit: bo versus bn counting\n\n'
    text+=markdown_table(['Dataset','bo total','bn total','bo zero items','bn zero items','bo zero but history>=5'],
        [[dataset,value['bo_total'],value['bn_total'],value['bo_zero_items'],value['bn_zero_items'],value['bo_zero_but_history_at_least5']]
         for dataset,value in background.items()])+'\n'
    text+='\n## Historical Real-Scope: separate item mapping and protocol\n\n'
    for dataset,value in historical.items():
        text+=f'\n### {dataset}\n\n'
        text+=value['compatibility']+' Statistics use the historical artifact background and training-frequency Head definition.\n\n'
        text+=markdown_table(['Level','Clusters','Size min-max','Mass mean','Mass SD','Mass min-max','Head-share SD'],
            [[v['level'],v['clusters'],f"{v['size_min']}-{v['size_max']}",f"{v['block_mass_mean']:.6f}",f"{v['block_mass_std']:.6f}",
              f"{v['block_mass_min']:.6f}-{v['block_mass_max']:.6f}",f"{v['head_fraction_std']:.6f}"] for v in value['levels']])+'\n'
    (ROOT/'diagnostics/report.md').write_text(text,encoding='utf-8')
    print(json.dumps({'background':background,'historical_real_scope':historical},indent=2))


if __name__=='__main__':
    main()
