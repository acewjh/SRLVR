"""Run three isolated base-eval lanes; save process state and notify on exit."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request


def save(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    tmp.replace(path)


def notify(cli, verdict):
    message = (f'[ALFWorld base full benchmark completion] report={cli.report}; '
               f'status={verdict}. Continue the original user task, inspect campaign.json '
               'and per-model summary.json and raw trajectories. Do not rerun successful evaluations.')
    try:
        result = subprocess.run(['/workspace/.local/bin/codex', 'queue', '--thread', cli.thread,
                                  '--message', message], capture_output=True, text=True, timeout=40)
        save(Path(cli.report)/'notification.json', dict(returncode=result.returncode,
                  stdout=result.stdout[-2000:], stderr=result.stderr[-2000:]))
    except Exception as exc:
        save(Path(cli.report)/'notification.json', dict(error=str(exc)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--thread', required=True)
    cli = parser.parse_args()
    report, project = Path(cli.report).resolve(), Path(cli.project).resolve()
    report.mkdir(parents=True, exist_ok=True)
    lock = (report/'campaign.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # This host's 8 cards were checked idle before launch. Refuse a new launch
    # if any accelerator process is now present; never terminate unrelated work.
    probe = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True, check=True)
    (report/'npu_before.txt').write_text(probe.stdout)
    if probe.stdout.count('No running processes found in NPU') != 8:
        raise RuntimeError('NPUs are occupied; refusing to overlap another job')
    for port in range(38200, 38206):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', port))
    snapshot = report/'source_snapshot'
    if not snapshot.exists():
        for rel in ('tasks/alfworld', 'opsd_core'):
            shutil.copytree(project/rel, snapshot/rel,
                ignore=shutil.ignore_patterns('__pycache__', 'data', 'upstream', 'tests', '*.bak.*'))
        for rel in ('tasks/__init__.py', 'tasks/alfworld_env_worker.py'):
            target = snapshot/rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(project/rel, target)
        (report/'git_diff.patch').write_text(subprocess.run(
            ['git', 'diff', '--', 'tasks/alfworld', 'opsd_core', 'frameworks/sglang/python/sglang/srt/layers/sampler.py'],
            cwd=project, capture_output=True, text=True, check=True).stdout)
        (report/'git_head.txt').write_text(subprocess.run(
            ['git', 'rev-parse', 'HEAD'], cwd=project, capture_output=True, text=True, check=True).stdout)
    model_root = project.parent.parent/'checkpoints'
    lanes = [('Qwen3-1.7B', ['0', '1'], 1, [38200, 38201]),
             ('Qwen3-4B', ['2', '3'], 1, [38202, 38203]),
             ('Qwen3-8B', ['4,5', '6,7'], 2, [38204, 38205])]
    state = dict(pid=os.getpid(), host=socket.gethostname(), start_time=time.time(),
                 status='running', models={}, thread_id=cli.thread)
    save(report/'campaign.json', state)
    processes = []

    def lane(model, devices, tp, ports):
        folder = report/model
        folder.mkdir(exist_ok=True)
        info = dict(model=str(model_root/model), tp=tp, devices=devices,
                    ports=ports, status='starting', server_pids=[], commands=[])
        # Every lane writes its own state, avoiding concurrent shared-file writes.
        servers = []
        try:
            for dev, port in zip(devices, ports):
                command = [sys.executable, '-m', 'sglang.launch_server', '--model-path', str(model_root/model),
                    '--tp', str(tp), '--port', str(port), '--host', '127.0.0.1', '--device', 'npu',
                    '--attention-backend', 'ascend', '--dtype', 'bfloat16', '--mem-fraction-static', '0.8',
                    '--context-length', '32768', '--chunked-prefill-size', '8192',
                    '--max-total-tokens', '65536', '--max-running-requests', '8',
                    '--cuda-graph-max-bs', '8', '--random-seed', '42',
                    '--nccl-port', str(port+1000)]
                socket_base = 40000+(port-38200)*100
                env = dict(os.environ, ASCEND_RT_VISIBLE_DEVICES=dev,
                    HCCL_HOST_SOCKET_PORT_RANGE=f'{socket_base}-{socket_base+49}',
                    HCCL_NPU_SOCKET_PORT_RANGE=f'{socket_base+1000}-{socket_base+1049}')
                log = (folder/f'engine_{port}.log').open('w')
                proc = subprocess.Popen(command, cwd=project, env=env, stdout=log,
                                        stderr=subprocess.STDOUT, start_new_session=True)
                log.close()
                servers.append(proc)
                processes.append(proc)
                info['server_pids'].append(proc.pid)
                info['commands'].append(command)
                save(folder/'lane.json', info)
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            deadline = time.monotonic()+900
            for port, proc in zip(ports, servers):
                while time.monotonic() < deadline:
                    if proc.poll() is not None:
                        raise RuntimeError(f'Engine {port} exited {proc.returncode}; inspect its log')
                    try:
                        with opener.open(f'http://127.0.0.1:{port}/health', timeout=3) as response:
                            if response.status == 200:
                                break
                    except Exception:
                        time.sleep(3)
                else:
                    raise RuntimeError(f'Engine {port} not ready within 900 seconds')
            for phase in ('smoke', 'full'):
                info['status'] = phase
                command = [sys.executable, '-m', 'tasks.alfworld.base_full_eval',
                    '--model', str(model_root/model), '--project', str(project),
                    '--engines', *[f'http://127.0.0.1:{port}' for port in ports],
                    '--out', str(folder/phase), '--concurrency', '8']
                if phase == 'smoke':
                    command.append('--smoke')
                info['commands'].append(command)
                with (folder/f'{phase}.log').open('a') as log:
                    client = subprocess.Popen(command, cwd=project, stdout=log,
                                              stderr=subprocess.STDOUT, start_new_session=True)
                    processes.append(client)
                    info['client_pid'] = client.pid
                    save(folder/'lane.json', info)
                    rc = client.wait()
                info[phase+'_returncode'] = rc
                if rc:
                    raise RuntimeError(f'{phase} returned {rc}; inspect {folder/phase}.log')
            info['status'] = 'complete'
        except Exception as exc:
            info.update(status='failed', error=str(exc))
        finally:
            # Stop only processes started by this lane, identified by retained PIDs.
            for proc in servers:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGTERM)
            for proc in servers:
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
            info['end_time'] = time.time()
            save(folder/'lane.json', info)
        return model, info

    def terminate(signum, frame):
        for proc in processes:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            for model, info in pool.map(lambda row: lane(*row), lanes):
                state['models'][model] = info
                save(report/'campaign.json', state)
        state['status'] = 'complete' if all(v['status'] == 'complete' for v in state['models'].values()) else 'failed'
    finally:
        state['end_time'] = time.time()
        save(report/'campaign.json', state)
        notify(cli, state['status'])
    return 0 if state['status'] == 'complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
