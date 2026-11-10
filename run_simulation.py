"""Provided background runner and result plotting helpers."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
import yaml
from assignment17 import TwoDimReservoir


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def process_token(pid):
    """Linux process start-time token; avoids treating a recycled PID as our run."""
    try:
        text = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if text[0] == 'Z' else text[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def simulate(directory, size, steps):
    started = time.monotonic()
    status = {'state': 'running', 'pid': os.getpid(), 'token': process_token(os.getpid()),
              'size': size, 'steps': steps, 'started_utc': datetime.now(timezone.utc).isoformat()}
    atomic_json(directory / 'status.json', status)
    try:
        inputs = yaml.safe_load(Path('inputs.yml').read_text())
        inputs['numerical']['number of grids'] = {'x': size, 'y': size}
        inputs['numerical']['number of time steps'] = steps
        inputs.pop('plots', None)  # store only final pressure, not every step
        model = TwoDimReservoir(inputs)
        assembled = time.monotonic()
        model.solve()
        pressure = model.get_solution().reshape(size, size)
        if not np.all(np.isfinite(pressure)):
            raise RuntimeError('simulation produced nonfinite pressure')
        np.savez_compressed(directory / 'pressure.npz', pressure=pressure,
                            length=inputs['reservoir']['length'], height=inputs['reservoir']['height'],
                            days=steps * inputs['numerical']['time step'])
        status.update(state='complete', elapsed_seconds=time.monotonic()-started,
                      assembly_seconds=assembled-started, pressure_min=float(pressure.min()),
                      pressure_max=float(pressure.max()))
    except BaseException as exc:
        status.update(state='failed', error=f'{type(exc).__name__}: {exc}')
        traceback.print_exc()
        raise
    finally:
        atomic_json(directory / 'status.json', status)
        print(json.dumps(status), flush=True)


def plot_result(directory):
    """Generate a labeled pressure map only from a completed simulation."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory = Path(directory)
    status = json.loads((directory / 'status.json').read_text())
    if status['state'] != 'complete':
        raise RuntimeError('cannot plot an incomplete simulation')
    with np.load(directory / 'pressure.npz') as result:
        pressure = result['pressure']
        if (pressure.ndim != 2 or pressure.size == 0 or
                not np.all(np.isfinite(pressure)) or
                any(not np.isfinite(float(result[key])) or float(result[key]) <= 0
                    for key in ('length', 'height', 'days'))):
            raise ValueError('invalid saved pressure field or plot metadata')
        fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
        x = (np.arange(pressure.shape[1]) + .5) * float(result['length']) / pressure.shape[1]
        y = (np.arange(pressure.shape[0]) + .5) * float(result['height']) / pressure.shape[0]
        image = ax.contourf(x, y, pressure, levels=20, cmap='viridis')
        ax.set_xlim(0, float(result['length']))
        ax.set_ylim(0, float(result['height']))
        ax.set_aspect('equal')
        fig.colorbar(image, ax=ax, label='Pressure (psi)')
        ax.scatter([5000], [5000], marker='^', color='white', edgecolors='black', label='Rate injector')
        ax.scatter([9000], [9000], marker='v', color='red', edgecolors='black', label='BHP producer')
        ax.set(xlabel='x (ft)', ylabel='y (ft)', title=f"Pressure at {float(result['days']):g} days")
        ax.legend(loc='lower left')
        temporary = directory / 'pressure.tmp.png'
        fig.savefig(temporary, dpi=150)
        plt.close(fig)
        temporary.replace(directory / 'pressure.png')


def launch(directory, size, steps):
    directory.mkdir(parents=True, exist_ok=False)  # never reuse another run's results
    atomic_json(directory / 'status.json', {'state': 'starting'})
    with (directory / 'simulation.log').open('w') as log:
        env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker',
                                    '--output', str(directory.resolve()), '--size', str(size),
                                    '--steps', str(steps)], stdout=log, stderr=log, env=env,
                                   start_new_session=True)
    print(f'Launched PID {process.pid}; results: {directory}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--size', type=int, default=1000)
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 3 <= args.size <= 1000 or not 1 <= args.steps <= 100:
        parser.error('size must be 3..1000 and steps 1..100')
    if args.worker:
        simulate(args.output, args.size, args.steps)
    else:
        launch(args.output, args.size, args.steps)
