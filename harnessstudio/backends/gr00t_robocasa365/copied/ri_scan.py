"""RI-2 render-integrity scanner (d3_render_noise_ruling.md, b49ea0d3...). Evaluator-owned, read-only.

For every results/<case>/episode.mp4 under the given run dirs (or every episode.mp4 under --tree):
  frame corrupt  iff  mean |horizontal neighbour difference| over RGB > 20  OR  per-frame std < 2
  episode flagged iff any corrupt frame, a decode error, or frame count != result.json 'chunks'
                      (also cross-checked against trace.jsonl action_dispatch + harness_inserted events)
Usage:
  python ri_scan.py --runs RUN [RUN ...] --out OUT.json        (per-arm check; exit 3 if any episode flagged)
  python ri_scan.py --tree DIR --out OUT.json                   (acceptance test over all session videos)
"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import cv2

HF_MAX, STD_MIN = 20.0, 2.0


def scan_video(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {'decode_error': 'open failed', 'frames': 0, 'corrupt_frames': [], 'hf': []}
    hf, std, corrupt = [], [], []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        f = frame.astype(np.float32)
        h = float(np.abs(f[:, 1:, :] - f[:, :-1, :]).mean())
        s = float(f.std())
        hf.append(round(h, 4)); std.append(round(s, 4))
        if h > HF_MAX or s < STD_MIN:
            corrupt.append(i)
        i += 1
    cap.release()
    return {'decode_error': None, 'frames': i, 'corrupt_frames': corrupt, 'hf': hf, 'std_min': min(std) if std else None}


def chunk_counts(case_dir):
    res = case_dir / 'result.json'
    chunks = json.loads(res.read_text()).get('chunks') if res.exists() else None
    tr = case_dir / 'trace.jsonl'
    tc = None
    if tr.exists():
        tc = sum(1 for l in tr.read_text().splitlines() if '"action_dispatch"' in l or '"harness_inserted"' in l)
    return chunks, tc


def scan_case(mp4):
    case_dir = mp4.parent
    v = scan_video(mp4)
    chunks, trace_chunks = chunk_counts(case_dir)
    count_ok = chunks is not None and v['frames'] == chunks and (trace_chunks is None or trace_chunks == chunks)
    flagged = bool(v['corrupt_frames']) or v['decode_error'] is not None or not count_ok
    return {'video': str(mp4), 'video_sha256': hashlib.sha256(mp4.read_bytes()).hexdigest(), 'frames': v['frames'],
            'result_chunks': chunks, 'trace_chunks': trace_chunks, 'frame_count_ok': count_ok,
            'decode_error': v['decode_error'], 'corrupt_frames': v['corrupt_frames'],
            'first_corrupt_frame': v['corrupt_frames'][0] if v['corrupt_frames'] else None,
            'n_corrupt': len(v['corrupt_frames']),
            'hf_max_clean': max([h for j, h in enumerate(v['hf']) if j not in v['corrupt_frames']], default=None),
            'hf_min_corrupt': min([v['hf'][j] for j in v['corrupt_frames']], default=None),
            'std_min': v['std_min'], 'flagged': flagged}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='*'); ap.add_argument('--tree'); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    vids = []
    if a.tree:
        vids = sorted(Path(a.tree).rglob('episode.mp4'))
    for r in a.runs or []:
        r = Path(r)
        m = json.loads((r / 'manifest.json').read_text())
        for c in m['cases']:
            d = r / 'results' / c['id']
            if (d / 'completed.json').exists():
                if not (d / 'episode.mp4').exists():
                    vids.append(d / 'episode.mp4')  # will be reported missing below
                else:
                    vids.append(d / 'episode.mp4')
    rows = []
    for v in vids:
        if not v.exists():
            rows.append({'video': str(v), 'flagged': True, 'decode_error': 'missing episode.mp4'})
        else:
            rows.append(scan_case(v))
    out = {'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'thresholds': {'hf_max': HF_MAX, 'std_min': STD_MIN},
           'videos': len(rows), 'flagged': [r['video'] for r in rows if r['flagged']],
           'max_clean_hf': max([r['hf_max_clean'] for r in rows if r.get('hf_max_clean') is not None], default=None),
           'per_video': rows}
    Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=True))
    print(json.dumps({'videos': out['videos'], 'flagged': [(r['video'], r.get('first_corrupt_frame'), r.get('n_corrupt'),
                      r.get('frame_count_ok'), r.get('decode_error')) for r in rows if r['flagged']], 'max_clean_hf': out['max_clean_hf']}, indent=1))
    return 3 if out['flagged'] else 0


if __name__ == '__main__':
    sys.exit(main())
