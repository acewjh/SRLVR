#!/usr/bin/env python3
"""Start one new run per worker, once; each run chains the existing benchmark."""
import argparse
import datetime
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[3]
SSH = ['ssh', '-o', 'StrictHostKeyChecking=no', '-o', 'UserKnownHostsFile=/dev/null',
       '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-p', '2222']
# Forward only experiment settings. Both nodes read one common environment file.
SETTINGS = '''MODEL NUM_LAYERS HIDDEN_SIZE FFN_HIDDEN_SIZE NUM_ATTENTION_HEADS NUM_QUERY_GROUPS
ACTOR_GPUS ROLLOUT_GPUS TP PP SEED OPSD_LORA_R OPSD_LORA_ALPHA OPSD_LORA_DROPOUT
LR EPS_CLIP EPS_CLIP_HIGH CLIP_GRAD MAX_TURNS MAX_RESPONSE HISTORY_LENGTH SEQ_LENGTH
SGLANG_PREFIX_CACHE SGLANG_DTYPE SGLANG_CHUNKED_PREFILL_SIZE ALFWORLD_THINKING_STYLE
ALFWORLD_ACTION_HARNESS ALFWORLD_TASK_SAMPLING ROLLOUT_BS ROLLOUT_TEMP ROLLOUT_TOP_P ROLLOUT_TOP_K
SRLVR_MODE SRLVR_CALIBRATION SRLVR_RHO SRLVR_WINDOW SRLVR_EPS_B0 SRLVR_DELTA_S
SRLVR_POSITION_CHUNK OPSD_HSI2_POS_CHUNK SAVE_INTERVAL SCHEDULE FUTURE_GAP EVAL_CONFIG
ALFWORLD_V2_EVAL_CONFIG N_SAMPLES_PER_EVAL_PROMPT ALFWORLD_EVAL_CONCURRENCY
MODEL_PROVIDER RECOMPUTE_NUM_LAYERS LOG_PROBS_CHUNK_SIZE RECOMPUTE_LOSS
SGLANG_RETURN_ORIGINAL_LOGPROB'''.split()


def remote(host, command):
    return subprocess.check_output(SSH + [host, command], text=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gamma', default=os.environ.get('SRLVR_GAMMA', '0.99'), type=float,
                        help='confirmed gamma, default 0.99')
    parser.add_argument('--check', action='store_true', help='check worker identities and device availability only')
    args = parser.parse_args()
    hosts = os.environ.get('VC_WORKER_HOSTS', '').split(',')
    if len(hosts) != 2 or not all(hosts):
        parser.error('VC_WORKER_HOSTS must identify exactly two workers')
    if not args.check and (args.gamma is None or not math.isfinite(args.gamma) or not 0 < args.gamma < 1):
        parser.error('set the agreed SRLVR_GAMMA (0 < gamma < 1) before launching')
    # Check both nodes before launching either run. Do not stop existing jobs.
    busy = []
    for host in hosts:
        identity = remote(host, 'hostname').strip()
        hardware = remote(host, '/usr/local/sbin/npu-smi info')
        pids = re.findall(r'^\|\s+\d+\s+\d+\s+(\d+)\s+\|', hardware, re.M)
        print(f'{identity}: accelerator_processes={len(pids)}', flush=True)
        if pids:
            busy.append(identity)
    if busy:
        raise SystemExit('Existing accelerator jobs on: ' + ', '.join(busy) + '. No new training was started.')
    if args.check:
        return
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    directory = ROOT / 'reports' / f'srlvr_v2_v3_{stamp}'
    directory.mkdir()
    common = {key: os.environ[key] for key in SETTINGS if key in os.environ}
    common['SRLVR_GAMMA'] = str(args.gamma)
    common['SRLVR_AUTO_BENCHMARK'] = '1'
    env_file = directory / 'common.env'
    env_file.write_text(''.join(f'export {key}={shlex.quote(value)}\n' for key, value in common.items()))
    manifest = dict(gamma=args.gamma, common_env=str(env_file), runs={})
    for host, variant in zip(hosts, ('v2', 'v3')):
        run = f'alfworld-qwen3-1p7b-srlvr_{variant}-s{common.get("SEED", "42")}-r25-{stamp}'
        run_root = ROOT / 'runs' / run
        log = directory / f'{variant}_launcher.log'
        script = ROOT / 'tasks/alfworld/scripts' / f'run_alfworld_srlvr_{variant}_1p7b.sh'
        command = (f'cd {shlex.quote(str(ROOT))} || exit\n'
            f'. {shlex.quote(str(env_file))} || exit\n'
            f'nohup env RUN_NAME={shlex.quote(run)} RUN_ROOT={shlex.quote(str(run_root))} '
            f'bash {shlex.quote(str(script))} > {shlex.quote(str(log))} 2>&1 < /dev/null &\n'
            'echo $!')
        # Only nohup is backgrounded, so no intermediate shell keeps SSH open.
        launcher_pid = remote(host, 'bash -c ' + shlex.quote(command)).strip()
        manifest['runs'][variant] = dict(host=host, run_name=run, run_root=str(run_root), launcher_log=str(log), launcher_pid=launcher_pid)
        (directory/'manifest.json').write_text(json.dumps(manifest, indent=2))
        print(f'Started {variant} on {host}: {run_root}', flush=True)


if __name__ == '__main__':
    main()
