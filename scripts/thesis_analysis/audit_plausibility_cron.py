"""Hourly audit of the v2 plausibility run: re-run randomly chosen finished cases independently and compare the verdict.
usage: audit_cron.py <workdir> <seed> [n]"""
import difflib, json, random, re, sys
from pathlib import Path
sys.path.insert(0, '/gpfs/helios/home/kandrei/PyTaRGET')
from data_processing.plausibility_test import (PlausibilityTester, iter_rows, project_row, find_repair_site, patch_source,
                                               common_margin, _parses)
from data_processing.metrics import normalize_indentation
R = '/gpfs/helios/home/kandrei/'; L = R + 'eval_results_perproject_llm/'
SETS = {'matched_qwen3coder': (L + 'matched/splits/test.json', L + 'matched/qwen3coder/test_predictions.json'),
        'matched_gptoss20b': (L + 'matched/splits/test.json', L + 'matched/gptoss20b/test_predictions.json'),
        'full_qwen3coder': (R + 'eval_results_perproject_fullctx/codet5p/0.8/splits/test.json', L + 'full/qwen3coder/test_predictions.json'),
        'full_gptoss20b': (R + 'eval_results_perproject_fullctx/codet5p/0.8/splits/test.json', L + 'full/gptoss20b/test_predictions.json')}
for _m in ('codet5p', 'plbart', 'codegen'):
    SETS['ft_' + _m] = (R + f'eval_results_perproject/{_m}/0.8/splits/test.json', R + f'eval_results_perproject/{_m}/0.8/test_predictions.json')
def result_file(name):
    return Path(R + (f'eval_results_perproject/{name[3:]}/0.8/plausibility_v3.json' if name.startswith('ft_')
                     else f'eval_llm_plaus/{name}/0.8/plausibility_v3.json'))
work, seed = Path(sys.argv[1]), int(sys.argv[2]); n = int(sys.argv[3]) if len(sys.argv) > 3 else 6
rnd = random.Random(seed)
pool = {'pass': [], 'fail': [], 'error': [], 'nondisc': []}
data = {}
for name, (tj, pj) in SETS.items():
    f = result_file(name)
    if not f.exists(): continue
    res = json.load(open(f))['results']
    p = json.load(open(pj)); preds = {p['ID'][k]: p['preds'][k] for k in p['ID']}
    data[name] = (res, preds)
    for k, v in res.items():
        if v['status'] != 'evaluated': continue
        vd = v['verdicts']
        if v.get('span_removed') == 'pass' and 'pass' in vd: pool['nondisc'].append((name, int(k), vd.index('pass')))
        elif 'pass' in vd: pool['pass'].append((name, int(k), vd.index('pass')))
        elif 'error' in vd: pool['error'].append((name, int(k), vd.index('error')))
        else: pool['fail'].append((name, int(k), 0))
print('finished evaluated cases:', {g: len(x) for g, x in pool.items()})
want = {'pass': 2, 'fail': 2, 'error': 1, 'nondisc': 1}
chosen = []
for g, c in want.items():
    chosen += [(g,) + x for x in rnd.sample(pool[g], min(c, len(pool[g])))]
need = {}
for g, name, i, r in chosen: need.setdefault(name, set()).add(i)
rows = {}
for name, ids in need.items():
    tj = SETS[name][0]
    for i, row in enumerate(iter_rows(tj)):
        if i in ids: rows[(name, i)] = project_row(row)
t = PlausibilityTester(work, k=5, timeout=120, workers=1)
bad = 0
summaries = []
for g, name, i, r in chosen:
    row = rows[(name, i)]; res, preds = data[name]; rec = res[str(i)]; cand = preds[i][r]
    line = f'[{g}] {name} id={i} {row["repository_name"]} rank={r}'
    summary_line = f'[{g}] {name} id={i} rank={r} recorded={rec["verdicts"][r]}'
    try:
        repo = t.prepare(row['repository_name'], row['repaired_hash']); python = t.python_for(row['repository_name'], repo)
        f = repo / row['relative_path']; orig = f.read_bytes(); src = orig.decode()
        site = find_repair_site(src, row['test_name'], row['output']); node = '::'.join([Path(row['relative_path']).as_posix()] + site.nodeid_parts)
        t.run_control(python, repo, node)   # like the real run: the control installs the known missing packages into the venv first
        new = patch_source(src, site, cand)
        a, b = src.split('\n'), new.split('\n')
        outside_ok = a[:site.start] == b[:site.start] and a[site.start + site.length:] == b[len(b) - (len(a) - site.start - site.length):]
        f.write_text(new, encoding='utf-8'); fresh = t.run_test(python, repo, node)
        span_removed = None
        if g == 'nondisc' or g == 'pass':
            removed = patch_source(src, site, 'pass')
            if not _parses(removed): span_removed = 'unparseable'      # like the pipeline: a span inside a statement is not run
            else: f.write_text(removed, encoding='utf-8'); span_removed = t.run_test(python, repo, node)
        f.write_bytes(orig)
        ok = (fresh == rec['verdicts'][r]) and outside_ok and node == rec['nodeid'] and (span_removed in (None, rec.get('span_removed')))
        bad += (not ok)
        nl = lambda s: len([x for x in s.strip('\n').split('\n') if x.strip()])
        has = lambda s: bool(re.search(r'\bassert\b|\.assert\w*\(|raises|pytest\.fail|expect', s))
        summary_line += f' fresh={fresh} recorded_span_removed={rec.get("span_removed")} fresh_span_removed={span_removed} {"OK" if ok else "MISMATCH"}'
        line += (f' -> nodeid_same={node == rec["nodeid"]} only_span_changed={outside_ok} parses={_parses(new)}'
                 f' cand_lines={nl(cand)} ref_lines={nl(row["output"])} cand_has_assert={has(cand)} ref_has_assert={has(row["output"])}'
                 f' => {"OK" if ok else "MISMATCH"}')
        # material for the self-appraisal: the test as patched with the candidate (the candidate's span marked), the
        # reference, and what pytest said
        sl = new.split('\n')[max(0, site.start - 6): site.start + max(len(cand.split('\n')), 1) + 8]
        line += '\n    CAND: ' + repr(cand[:400]) + '\n    REF : ' + repr(row['output'][:400])
        line += '\n    PATCHED TEST (around the span, span starts after 6 context lines):\n' + '\n'.join('      | ' + x for x in sl[:40])
        if fresh != 'pass':
            line += '\n    PYTEST TAIL: ' + t.last_output[-350:].replace('\n', ' | ')
    except Exception as e:
        line += f' -> ERROR {e!r}'; bad += 1; summary_line += f' ERROR {e!r}'
    print(line, flush=True)
    summaries.append(summary_line)
print('\n=== VERDICTS (read after your own appraisal) ===')
print('\n'.join(summaries))
print('MISMATCHES:', bad, 'of', len(chosen))
