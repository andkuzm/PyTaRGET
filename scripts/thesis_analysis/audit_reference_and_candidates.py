"""Audit of the plausibility scoring: re-run chosen cases through the real PlausibilityTester and probe each with
controls (reference, candidate, span deleted). Output: ~/eval_audit/report.json + diffs."""
import difflib, json, random, re, sys
from pathlib import Path
sys.path.insert(0, '/gpfs/helios/home/kandrei/PyTaRGET')
from data_processing.plausibility_test import (PlausibilityTester, iter_rows, project_row, find_repair_site, patch_source,
                                               common_margin)
from data_processing.metrics import normalize_indentation
R = '/gpfs/helios/home/kandrei/'
OUT = Path(R + 'eval_audit'); OUT.mkdir(exist_ok=True)
M = 'gptoss20b'
pl = json.load(open(R + f'eval_llm_plaus/full_{M}/0.8/plausibility.json'))['results']
p = json.load(open(R + f'eval_results_perproject_llm/full/{M}/test_predictions.json'))
preds = {p['ID'][k]: p['preds'][k] for k in p['ID']}; refs = {p['ID'][k]: p['target'][k] for k in p['ID']}
nl = lambda s: [l for l in s.strip('\n').split('\n') if l.strip()]
hasassert = lambda s: bool(re.search(r'\bassert\b|\.assert\w*\(|raises|pytest\.fail|\bexpect', s))
rnd = random.Random(11)
groups = {'pass_short': [], 'pass_noassert': [], 'pass_other': [], 'all_fail': [], 'has_error': []}
for k, v in pl.items():
    if v['status'] != 'evaluated': continue
    i = int(k); vd = v['verdicts']; ref = refs[i]
    if 'pass' in vd:
        r = vd.index('pass'); c = preds[i][r]
        if normalize_indentation(c).strip() == normalize_indentation(ref).strip(): continue
        if hasassert(ref) and not hasassert(c): groups['pass_noassert'].append(i)
        elif len(nl(c)) < len(nl(ref)): groups['pass_short'].append(i)
        else: groups['pass_other'].append(i)
    elif 'error' in vd: groups['has_error'].append(i)
    else: groups['all_fail'].append(i)
print({g: len(x) for g, x in groups.items()}, flush=True)
want = {'pass_short': 4, 'pass_noassert': 4, 'pass_other': 3, 'all_fail': 3, 'has_error': 2}
chosen = [(g, i) for g, n in want.items() for i in rnd.sample(groups[g], n)]
ids = {i for _, i in chosen}
rows = {}
for i, row in enumerate(iter_rows(R + 'eval_results_perproject_fullctx/codet5p/0.8/splits/test.json')):
    if i in ids: rows[i] = project_row(row)
t = PlausibilityTester(Path(sys.argv[1]), k=5, timeout=120, workers=1)
report = []
for g, i in chosen:
    row = rows[i]; rec = pl[str(i)]; cands = preds[i]
    entry = {'group': g, 'id': i, 'repo': row['repository_name'], 'nodeid': rec['nodeid'], 'recorded': rec['verdicts']}
    try:
        fresh = t.evaluate_case(row, cands)
        entry['fresh'] = fresh['verdicts']; entry['fresh_status'] = fresh['status']
        repo = t.prepare(row['repository_name'], row['repaired_hash'])
        f = repo / row['relative_path']; orig = f.read_bytes()
        src = orig.decode(); site = find_repair_site(src, row['test_name'], row['output'])
        python = t.python_for(row['repository_name'], repo)
        node = rec['nodeid']
        margin = common_margin(src.split('\n')[site.start:site.start + site.length])
        variants = {'reference': row['output'], 'span_replaced_by_pass': 'pass'}
        r0 = rec['verdicts'].index('pass') if 'pass' in rec['verdicts'] else 0
        variants['candidate'] = cands[r0]
        for name, text in variants.items():
            new = patch_source(src, site, text)
            f.write_text(new, encoding='utf-8')
            entry[name] = t.run_test(python, repo, node)
            if name == 'candidate':
                diff = ''.join(difflib.unified_diff(src.splitlines(1), new.splitlines(1), 'repaired_commit', 'with_candidate', n=2))
                (OUT / f'{g}_{i}.diff').write_text(diff)
                entry['candidate_text'] = cands[r0][:600]; entry['reference_text'] = row['output'][:600]
                entry['pytest_tail'] = t.last_output
        f.write_bytes(orig)
    except Exception as e:
        entry['error'] = repr(e)
    print(g, i, entry.get('recorded'), '->fresh', entry.get('fresh'), '| ref', entry.get('reference'), 'cand', entry.get('candidate'),
          'span->pass', entry.get('span_replaced_by_pass'), flush=True)
    report.append(entry)
json.dump(report, open(OUT / 'report.json', 'w'), indent=1)
