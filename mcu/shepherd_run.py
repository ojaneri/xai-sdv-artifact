#!/usr/bin/env python3
"""Submit an ELF to Shepherd Nova, wait, download, hash. Retries on failure.

Adapted from an earlier submission script of ours (same client calls, same
retry policy). Credentials come from the client's saved
config (~/.config/shepherd/client.yaml); nothing secret is read or written here.

    /opt/shepherd-venv/bin/python shepherd_run.py build/xai_nrf52840.elf \
        --duration 900 --target 3
"""
import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('elf')
    ap.add_argument('--duration', type=int, default=900)
    ap.add_argument('--target', type=int, default=3)
    ap.add_argument('--source', default='direct')
    ap.add_argument('--env', default='synthetic_static_3000mV_50mA')
    ap.add_argument('--name', default=None)
    a = ap.parse_args()

    from shepherd_client import Client
    from shepherd_core.data_models.content import Firmware
    from shepherd_core.data_models.experiment import Experiment, TargetConfig

    elf = Path(a.elf).resolve()
    name = a.name or elf.stem
    out = HERE / 'results' / name
    out.mkdir(parents=True, exist_ok=True)
    c = Client()

    exp_id = None
    for attempt in range(1, 6):
        try:
            tc = TargetConfig(
                target_IDs=[a.target],
                energy_env=c.get_resource_item('EnergyEnvironment', name=a.env),
                virtual_source=c.get_resource_item('VirtualSourceConfig', name=a.source),
                firmware1=Firmware.from_firmware(elf, name=elf.stem),
                uart_logging={'baudrate': 115200},
                gpio_tracing={'gpios': [0, 1, 2, 3, 4, 5]},
                power_tracing={'samplerate': 100000},
            )
            xp = Experiment(name='xai-' + name,
                            description='XAI explanation cost on nRF52840 (XAI-SDV-E)',
                            duration=timedelta(seconds=a.duration),
                            target_configs=[tc])
            exp_id = c.create_experiment(xp=xp)
            if not exp_id:
                raise RuntimeError('create_experiment returned no id')
            c.schedule_experiment(exp_id)
            break
        except Exception as e:  # lab is flaky by the operator's own warning
            print(f'attempt {attempt} failed: {e}', flush=True)
            time.sleep(30)
    if not exp_id:
        sys.exit('all submission attempts failed')

    meta = {'experiment_id': str(exp_id), 'platform': 'shepherd-nova',
            'mcu': 'nRF52840 @ 64 MHz', 'target_id': a.target,
            'duration_s': a.duration, 'virtual_source': a.source,
            'energy_environment': a.env, 'firmware': elf.name,
            'firmware_sha256': sha256(elf),
            'submitted_at': datetime.now(timezone.utc).isoformat()}
    (out / 'experiment.json').write_text(json.dumps(meta, indent=1))
    print('experiment', exp_id, flush=True)

    terminal = {'done', 'finished', 'stopped', 'error', 'failed'}
    while True:
        st = str(c.get_experiment_state(exp_id))
        print(time.strftime('%H:%M:%S'), 'state', st, flush=True)
        if st.lower() in terminal:
            break
        time.sleep(30)
    meta['final_state'] = st
    ok = c.download_experiment(exp_id, out)
    meta['downloaded'] = bool(ok)
    meta['files'] = {str(p.relative_to(out)): sha256(p)
                     for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'experiment.json'}
    (out / 'experiment.json').write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == '__main__':
    main()
