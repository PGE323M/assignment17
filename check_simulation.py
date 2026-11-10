"""Provided one-shot completion check for an agent scheduler or cron.

This module never schedules work or launches a simulation. Exit codes are
0 complete, 1 failed, and 2 running/busy. The caller owns task removal.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import time
from run_simulation import atomic_json, plot_result, process_token


def check_status(directory):
    """Read one atomic run-status snapshot; raise on invalid/failed work."""
    status_path = directory / 'status.json'
    status = json.loads(status_path.read_text())
    state = status['state']
    if state == 'starting':
        if time.time() - status_path.stat().st_mtime > 60:
            raise RuntimeError('simulation did not start within 60 seconds')
        return 'running'
    if state == 'running':
        token = status.get('token')
        if token is None or process_token(status['pid']) != token:
            raise RuntimeError('simulation process is missing, dead, or recycled')
        return 'running'
    if state == 'failed':
        raise RuntimeError(status.get('error', 'simulation failed; inspect simulation.log'))
    if state != 'complete':
        raise ValueError(f'unknown simulation state {state!r}')
    return 'complete'


def check_run(directory):
    """Check once, serialize concurrent checks, and save a terminal outcome."""
    directory = Path(directory)
    if not directory.is_dir():
        return {'state': 'failed', 'error': f'run directory does not exist: {directory}'}
    with (directory / 'completion.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {'state': 'running', 'detail': 'another completion check is active'}
        outcome = {'checked_utc': datetime.now(timezone.utc).isoformat()}
        try:
            marker = directory / 'completion.json'
            if marker.exists():
                previous = json.loads(marker.read_text())
                if previous['state'] not in ('complete', 'failed'):
                    raise ValueError('invalid terminal completion record')
                if previous['state'] == 'complete':
                    if (directory / 'pressure.png').read_bytes()[:8] != b'\x89PNG\r\n\x1a\n':
                        raise ValueError('completed plot is missing or invalid')
                return previous
            outcome['state'] = check_status(directory)
            if outcome['state'] == 'running':
                return outcome
            # A unique run directory prevents stale results from another run.
            # Existing PNG without completion marker is an interrupted attempt:
            # regenerate it from validated saved results, then commit the marker.
            plot_result(directory)
            if (directory / 'pressure.png').read_bytes()[:8] != b'\x89PNG\r\n\x1a\n':
                raise ValueError('plot helper did not produce a PNG')
        except Exception as exc:
            outcome.update(state='failed', error=f'{type(exc).__name__}: {exc}')
            # Never leave a success-looking plot after terminal validation fails.
            (directory / 'pressure.png').unlink(missing_ok=True)
            (directory / 'pressure.tmp.png').unlink(missing_ok=True)
        atomic_json(directory / 'completion.json', outcome)
        return outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        outcome = check_run(args.output)
    except Exception as exc:
        outcome = {'state': 'failed', 'error': f'{type(exc).__name__}: {exc}'}
    print(json.dumps(outcome), flush=True)
    return {'complete': 0, 'failed': 1, 'running': 2}[outcome['state']]


if __name__ == '__main__':
    raise SystemExit(main())
