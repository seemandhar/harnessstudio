"""NP v2: null-vs-parent per-case M/R/X first-divergence attribution (null_precheck_standard_ruling.md section 2).

v2 = v1 (ae805062...) + runner_v2_2_review.md NP-F1 (firings read from a declared per-case candidate log,
cross-checked against a declared result.json counter; '--firing-event none' only with a candidate-freeze
declaration) + D-2 (benign render-noise blocks flagged as infrastructure) + D-3 (per-phase b and q).

Evaluator-owned, outcome-blind in its routing except the M condition "outcomes equal" (NP-1).
Generalizes c2_escape_select/logs/evaluator/fix2_crosscheck_rev1.py (d4c471c4...). Read-only
on sealed records; writes ONE write-once JSON.

NP-1  per case (null vs parent named in the candidate freeze), action_dispatch rows compared on
      input (per key), action, noise_seed, execute_steps and selected_k where present:
        M  every row identical, equal row counts, equal outcomes
        R  first differing row: all earlier rows identical; noise_seed, execute_steps and every
           state.*/annotation key equal; >= 1 video.* key differs; RI-2 clean in BOTH runs
        X  anything else (incl. equal trajectory with unequal outcome; row-count difference
           without an earlier R; missing/incomplete records)
NP-2  pass iff 0 X, R <= 20 of 32, RI clean on all three reference arms, gate_v1_2
      null_precheck route == proceed_to_candidate on the same arm manifests
NP-3  any failure -> infrastructure_blocked (reason recorded)
NP-4  firing coverage: firings are read from results/<case>/<--firing-log> rows whose 'event' equals
      --firing-event (chunk from --firing-chunk-field); per case the count must equal
      result.json[--firing-count-field] (missing log/counter or mismatch -> blocked). If the null fires,
      at least one firing must lie inside a verified (identical) prefix. '--firing-event none' requires
      the candidate freeze to declare "null_path_every_decision": true; coverage is then >= 1 verified
      prefix chunk in at least one case.
D-2   blocked ONLY by R > 20 and/or the gate's outcome-level null limits, with 0 X and RI clean ->
      infrastructure_blocked (render noise) -> critic ruling with measured b, q; no reference re-draw
D-3   reports b = R/cases and q = (R pairs with unequal outcomes)/R every phase

  python np_attribution_v1.py --parent PARENT_OUT --null NULL_OUT --other-ref OTHER_REF_OUT \
     --ri-parent RI.json --ri-null RI.json --ri-other RI.json --precheck GATE_OUT/null_precheck.json \
     --candidate-freeze CF.json --firing-log dwell_hook.jsonl --firing-event shadow_fire \
     --firing-count-field shadow_activations --out results/development-v2_2/np_attribution.json
  Acceptance mode (C2 census pair, NP-5): add --acceptance (counts only, no routing; no precheck/other).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

FIELDS = ('input', 'action', 'noise_seed', 'execute_steps', 'selected_k')
R_MAX = 20


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def trace(run, case):
    path = Path(run) / 'results' / case / 'trace.jsonl'
    return [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else None


def ri_flags(path):
    """RI-2 report {'per_video': [{'video': '.../results/CASE/...', 'flagged': bool}]} -> {case: flagged}."""
    if path is None:
        return None
    rows = json.loads(Path(path).read_text())['per_video']
    return {r['video'].rsplit('/results/', 1)[1].split('/')[0]: bool(r['flagged']) for r in rows}


def firing_chunks(null, case, firing):
    """Firings from the declared per-case candidate log; None when the log is missing."""
    if not firing or firing.get('event') in (None, 'none'):
        return []
    if not firing.get('log'):
        return None
    path = Path(null) / 'results' / case / firing['log']
    if not path.is_file():
        return None
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return [r.get(firing['chunk_field']) for r in rows if r.get('event') == firing['event']]


def classify(case, parent, null, ri_parent, ri_null, firing):
    tp, tn = trace(parent, case), trace(null, case)
    sealed = all((Path(run) / 'results' / case / 'completed.json').exists() for run in (parent, null))
    if tp is None or tn is None or not sealed:
        return {'class': 'X', 'reason': 'missing or unsealed record'}
    a = [r for r in tp if r.get('event') == 'action_dispatch']
    b = [r for r in tn if r.get('event') == 'action_dispatch']
    first = next((i for i, (x, y) in enumerate(zip(a, b)) if any(x.get(f) != y.get(f) for f in FIELDS)), None)
    out_p = json.loads((Path(parent) / 'results' / case / 'result.json').read_text()).get('success')
    null_result = json.loads((Path(null) / 'results' / case / 'result.json').read_text())
    out_n = null_result.get('success')
    firings = firing_chunks(null, case, firing)
    crosscheck = None
    if firing and firing.get('event') not in (None, 'none'):
        counter = null_result.get(firing['count_field'])
        crosscheck = ('missing_log' if firings is None else 'missing_counter' if type(counter) is not int
                      else 'ok' if counter == len(firings) else f'mismatch: log {len(firings)} != counter {counter}')
    firings = firings or []
    info = {'rows': [len(a), len(b)], 'null_firings': len(firings), 'firing_crosscheck': crosscheck}
    if first is None:
        if len(a) != len(b):
            return dict(info, **{'class': 'X', 'reason': 'row-count difference without an earlier R'})
        if out_p != out_n:
            return dict(info, **{'class': 'X', 'reason': 'equal trajectory, unequal outcome'})
        return dict(info, **{'class': 'M', 'verified_prefix_chunks': len(a),
                             'null_firings_in_prefix': len(firings)})
    x, y = a[first], b[first]
    xi, yi = x.get('input') or {}, y.get('input') or {}
    keys = set(xi) | set(yi)
    state_eq = all(xi.get(k) == yi.get(k) for k in keys if k.startswith('state.') or k.startswith('annotation'))
    video_diff = sorted(k for k in keys if k.startswith('video.') and xi.get(k) != yi.get(k))
    ri_clean = (ri_parent is not None and ri_null is not None
                and ri_parent.get(case) is False and ri_null.get(case) is False)
    detail = {'first_divergent_chunk': first, 'video_keys_differ': video_diff, 'state_equal': state_eq,
              'seed_equal': x.get('noise_seed') == y.get('noise_seed'),
              'execute_equal': x.get('execute_steps') == y.get('execute_steps'),
              'action_equal_at_first': x.get('action') == y.get('action'), 'ri_clean_both': ri_clean}
    is_r = (detail['seed_equal'] and detail['execute_equal'] and state_eq and bool(video_diff) and ri_clean)
    in_prefix = sum(1 for c in firings if c is not None and c < first)
    return dict(info, **{'class': 'R' if is_r else 'X', 'detail': detail, 'verified_prefix_chunks': first,
                         'null_firings_in_prefix': in_prefix,
                         'outcomes_equal': out_p == out_n})


def run(parent, null, *, cases, ri_parent=None, ri_null=None, ri_other=None, other_ref=None, precheck=None,
        firing_event=None, acceptance=False, firing_log=None, firing_chunk_field='chunk', firing_count_field=None,
        candidate_freeze=None):
    fp, fn, fo = ri_flags(ri_parent), ri_flags(ri_null), ri_flags(ri_other)
    firing = None if firing_event is None else {'event': firing_event, 'log': firing_log,
                                                'chunk_field': firing_chunk_field, 'count_field': firing_count_field}
    res = {c: classify(c, parent, null, fp, fn, firing) for c in cases}
    if acceptance:  # NP-5 census acceptance (as fix2_crosscheck_rev1): incomplete cases are excluded, not X
        for v in res.values():
            if v.get('reason') == 'missing or unsealed record':
                v['class'] = 'excluded_incomplete'
    counts = {k: sum(v['class'] == k for v in res.values()) for k in ('M', 'R', 'X', 'excluded_incomplete')}
    firings = sum(v.get('null_firings', 0) for v in res.values())
    in_prefix = sum(v.get('null_firings_in_prefix', 0) for v in res.values())
    r_cases = [v for v in res.values() if v['class'] == 'R']
    discordant_r = sum(1 for v in r_cases if v.get('outcomes_equal') is False)
    summary = {'counts': counts, 'cases': len(cases), 'null_firings_total': firings,
               'null_firings_in_verified_prefix': in_prefix, 'firing': firing,
               'calibration_D3': {'b': counts['R'] / len(cases) if cases else None, 'R': counts['R'],
                                  'discordant_R_pairs': discordant_r,
                                  'q': (discordant_r / counts['R']) if counts['R'] else None},
               'first_divergence': {c: v['detail']['first_divergent_chunk'] for c, v in res.items() if 'detail' in v}}
    if acceptance:
        summary['routing'] = 'acceptance_counts_only'
        return {'summary': summary, 'cases': res}
    reasons = []
    if counts['X']:
        reasons.append('X present (suspected seam non-neutrality) -> candidate owner')
    if counts['R'] > R_MAX:
        reasons.append(f'R {counts["R"]} > {R_MAX} (render-rate anomaly) -> critic ruling')
    for label, flags, arm in (('parent', fp, parent), ('null', fn, null), ('other_reference', fo, other_ref)):
        if flags is None or arm is None:
            reasons.append(f'RI report missing for {label}')
        elif any(flags.get(c) is not False for c in cases):
            reasons.append(f'RI not clean (or case missing) for {label}')
    if firing_event is None:
        reasons.append('--firing-event must be declared (NP-F1)')
    elif firing_event == 'none':
        cf = json.loads(Path(candidate_freeze).read_text()) if candidate_freeze else {}
        if cf.get('null_path_every_decision') is not True:
            reasons.append('--firing-event none requires candidate freeze null_path_every_decision: true (NP-F1)')
        elif max((v.get('verified_prefix_chunks') or 0) for v in res.values()) < 1:
            reasons.append('no verified prefix chunk in any case (NP-4, every-decision null)')
    else:
        if not firing_log or not firing_count_field:
            reasons.append('--firing-log and --firing-count-field are required with a firing event (NP-F1)')
        bad = {c: v['firing_crosscheck'] for c, v in res.items() if v.get('firing_crosscheck') != 'ok'}
        if bad:
            reasons.append(f'firing log/counter cross-check failed in {len(bad)} case(s) (NP-F1)')
            summary['firing_crosscheck_failures'] = bad
        if firings > 0 and in_prefix == 0:
            reasons.append('null fires but no firing lies inside a verified prefix (NP-4)')
    manifests = {'parent': sha256(Path(parent) / 'manifest.json'), 'null': sha256(Path(null) / 'manifest.json')}
    if other_ref is not None:
        manifests['other_reference'] = sha256(Path(other_ref) / 'manifest.json')
    if precheck is None:
        reasons.append('gate null_precheck missing')
    else:
        pre = json.loads(Path(precheck).read_text())
        if pre.get('route') != 'proceed_to_candidate':
            reasons.append('gate null_precheck route is not proceed_to_candidate')
        arms = pre.get('inputs', {}).get('arms', {})
        names = {'parent': pre.get('parent'), 'null': 'null'}
        for key, arm_name in names.items():
            if arms.get(arm_name, {}).get('manifest_sha256') != manifests[key]:
                reasons.append(f'gate null_precheck was computed on a different {key} arm')
        manifests['precheck'] = sha256(precheck)
    benign_only = {'render_rate'}
    kinds = set()
    for reason in reasons:
        if reason.startswith('R ') and '> 20' in reason:
            kinds.add('render_rate')
        elif reason == 'gate null_precheck route is not proceed_to_candidate' and precheck is not None:
            pre = json.loads(Path(precheck).read_text())
            nc = pre.get('null_control', {})
            limits_only = (not pre.get('blocked_arms') and nc.get('complete') and nc.get('comparable')
                           and pre.get('noise_ok') is False)
            kinds.add('outcome_null_limits' if limits_only else 'other')
        else:
            kinds.add('other')
    benign = bool(reasons) and counts['X'] == 0 and kinds <= {'render_rate', 'outcome_null_limits'}
    summary.update(reasons=reasons, routing='proceed' if not reasons else 'infrastructure_blocked',
                   passed=not reasons, input_sha256=manifests,
                   benign_render_noise_block_D2=benign,
                   route_note=('infrastructure_blocked (render noise, D-2): critic ruling with measured b, q; '
                               'no reference arm re-drawn; not negative evidence') if benign else None)
    return {'summary': summary, 'cases': res}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--parent', required=True); ap.add_argument('--null', required=True)
    ap.add_argument('--other-ref'); ap.add_argument('--ri-parent'); ap.add_argument('--ri-null')
    ap.add_argument('--ri-other'); ap.add_argument('--precheck'); ap.add_argument('--firing-event')
    ap.add_argument('--firing-log'); ap.add_argument('--firing-chunk-field', default='chunk')
    ap.add_argument('--firing-count-field'); ap.add_argument('--candidate-freeze')
    ap.add_argument('--acceptance', action='store_true'); ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)
    manifest = json.loads((Path(args.null) / 'manifest.json').read_text())
    cases = [c['id'] for c in manifest['cases']]
    result = run(args.parent, args.null, cases=cases, ri_parent=args.ri_parent, ri_null=args.ri_null,
                 ri_other=args.ri_other, other_ref=args.other_ref, precheck=args.precheck,
                 firing_event=args.firing_event, acceptance=args.acceptance, firing_log=args.firing_log,
                 firing_chunk_field=args.firing_chunk_field, firing_count_field=args.firing_count_field,
                 candidate_freeze=args.candidate_freeze)
    result['summary']['script_sha256'] = sha256(__file__)
    fd = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(fd, 'w') as stream:
        json.dump(result, stream, indent=1, sort_keys=True)
    print(json.dumps({k: result['summary'].get(k) for k in ('counts', 'routing', 'reasons')}, indent=1))
    return 0 if result['summary'].get('routing') in ('proceed', 'acceptance_counts_only') else 3


if __name__ == '__main__':
    raise SystemExit(main())
