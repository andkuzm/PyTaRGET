import json,re,sys
sys.path.insert(0,'/gpfs/helios/home/kandrei/PyTaRGET')
from data_processing.plausibility_test import iter_rows
from data_processing.metrics import normalize_indentation
R='/gpfs/helios/home/kandrei/'
refs={}
for i,row in enumerate(iter_rows(R+'eval_results_perproject_fullctx/codet5p/0.8/splits/test.json')):
    refs[i]=row['output']
def nl(s): return [l for l in s.strip('\n').split('\n') if l.strip()]
def hasassert(s): return bool(re.search(r'\bassert\b|\.assert\w*\(|raises|pytest\.fail|\bexpect', s))
for m in ('qwen3coder','gptoss20b'):
    pl=json.load(open(R+f'eval_llm_plaus/full_{m}/0.8/plausibility.json'))['results']
    pr={r['ID']:r['preds'] for r in json.load(open(R+f'eval_results_perproject_llm/full/{m}/test_predictions.json'))} if False else None
    import pandas as pd
    p=json.load(open(R+f'eval_results_perproject_llm/full/{m}/test_predictions.json'))
    preds={p['ID'][k]:p['preds'][k] for k in p['ID']}
    n=same=shorter=noassert_cand_ref_assert=eq=0
    for k,v in pl.items():
        if v['status']!='evaluated': continue
        for r,(c,vd) in enumerate(zip(preds[int(k)][:5],v['verdicts'])):
            if vd!='pass': continue
            n+=1; ref=refs[int(k)]
            if normalize_indentation(c).strip()==normalize_indentation(ref).strip(): eq+=1; continue
            if len(nl(c))<len(nl(ref)): shorter+=1
            if hasassert(ref) and not hasassert(c): noassert_cand_ref_assert+=1
    print(m,'passing candidates',n,'equal to reference',eq,'fewer non-empty lines than reference',shorter,'reference has assert but candidate none',noassert_cand_ref_assert)
