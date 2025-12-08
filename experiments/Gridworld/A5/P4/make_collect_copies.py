import json
import os
import glob
from pathlib import Path

SOURCE_PATH = 'experiments/Gridworld/A5/P4/gridworldpartial_transfer'
SRC = Path(SOURCE_PATH)
DST = Path('experiments/Gridworld/A5/P4/gridworldpartial_transfer_collect')


def ensure(d: dict, *keys):
    """Ensure nested dict keys exist and return the final dict."""
    cur = d
    for k in keys:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    return cur


def process_file(src_path: Path, dst_path: Path):
    with src_path.open('r') as f:
        data = json.load(f)

    data['total_steps'] = 100_000

    meta = data.setdefault('metaParameters', {})

    meta['update_freq'] = 1_000_000
    meta['buffer_size'] = 100_000

    opt = meta.setdefault('optimizer', {})
    opt['alpha'] = 0.0

    exp = meta.setdefault('experiment', {})
    load = exp.setdefault('load', {})
    cfg = load.setdefault('config', {})
    a = cfg.setdefault('a', {})
    state = a.setdefault('state', {})
    params = state.setdefault('params', {})
    params['q'] = True
    target = state.setdefault('target_params', {})
    target['q'] = True

    agent = data.get('agent', '')
    
    load['path'] = f"{SOURCE_PATH.replace('experiments', 'results')}/{agent}"

    # write output
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with dst_path.open('w') as f:
        json.dump(data, f, indent=4)

    print(f'WROTE {dst_path}')


def main():
    if not SRC.exists():
        print(f'ERROR: source folder does not exist: {SRC}')
        return
    DST.mkdir(parents=True, exist_ok=True)

    files = sorted(SRC.glob('*.json'))
    if not files:
        print('No JSON files found in', SRC)
        return

    for p in files:
        dst_file = DST / p.name
        try:
            process_file(p, dst_file)
        except Exception as e:
            print('SKIP', p, '-> error:', e)

    print(f'Done. Processed {len(files)} files.')


if __name__ == '__main__':
    main()
