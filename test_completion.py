"""Checks for provided single-invocation status/plot infrastructure."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
import check_simulation as checker
from run_simulation import atomic_json, process_token


class CompletionChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def status(self, **values):
        atomic_json(self.directory/'status.json', values)

    def results(self):
        np.savez_compressed(self.directory/'pressure.npz',pressure=np.full((3,3),1000.),
                            length=10000.,height=10000.,days=3.)

    def test_running_and_recycled_process(self):
        self.status(state='running',pid=os.getpid(),token=process_token(os.getpid()))
        self.assertEqual(checker.check_run(self.directory)['state'],'running')
        self.assertFalse((self.directory/'completion.json').exists())
        self.status(state='running',pid=os.getpid(),token='recycled')
        result=checker.check_run(self.directory)
        self.assertEqual(result['state'],'failed')
        self.assertIn('recycled',result['error'])

    def test_success_plots_once_and_terminal_is_idempotent(self):
        self.status(state='complete');self.results()
        with patch('check_simulation.plot_result',wraps=checker.plot_result) as plot:
            first=checker.check_run(self.directory)
            second=checker.check_run(self.directory)
            self.assertEqual(first,second)
            self.assertEqual(plot.call_count,1)
        self.assertEqual(first['state'],'complete')
        self.assertTrue((self.directory/'pressure.png').read_bytes().startswith(b'\x89PNG'))

    def test_failed_run_never_plots(self):
        self.status(state='failed',error='test simulation failure')
        with patch('check_simulation.plot_result') as plot:
            result=checker.check_run(self.directory)
            plot.assert_not_called()
        self.assertEqual(result['state'],'failed')
        self.assertIn('test simulation failure',result['error'])

    def test_stale_start_and_partial_status(self):
        self.status(state='starting')
        self.assertEqual(checker.check_run(self.directory)['state'],'running')
        os.utime(self.directory/'status.json',(time.time()-65,time.time()-65))
        self.assertEqual(checker.check_run(self.directory)['state'],'failed')
        (self.directory/'completion.json').unlink()
        (self.directory/'status.json').write_text('{partial')
        self.assertEqual(checker.check_run(self.directory)['state'],'failed')

    def test_missing_or_nonfinite_results_fail_without_plot(self):
        self.status(state='complete')
        self.assertEqual(checker.check_run(self.directory)['state'],'failed')
        self.assertFalse((self.directory/'pressure.png').exists())
        (self.directory/'completion.json').unlink()
        np.savez_compressed(self.directory/'pressure.npz',pressure=np.full((3,3),np.nan),
                            length=10000.,height=10000.,days=3.)
        self.assertEqual(checker.check_run(self.directory)['state'],'failed')
        self.assertFalse((self.directory/'pressure.png').exists())

    def test_concurrent_check_does_not_duplicate_plot(self):
        self.status(state='complete');self.results()
        with (self.directory/'completion.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            result=checker.check_run(self.directory)
        self.assertEqual(result['state'],'running')
        self.assertIn('another',result['detail'])
        self.assertFalse((self.directory/'pressure.png').exists())
        self.assertFalse((self.directory/'completion.json').exists())

    def test_interrupted_plot_is_regenerated(self):
        self.status(state='complete');self.results()
        (self.directory/'pressure.png').write_bytes(b'partial png')
        self.assertEqual(checker.check_run(self.directory)['state'],'complete')
        self.assertTrue((self.directory/'pressure.png').read_bytes().startswith(b'\x89PNG'))

    def test_terminal_plot_corruption_is_failure(self):
        self.status(state='complete');self.results()
        self.assertEqual(checker.check_run(self.directory)['state'],'complete')
        (self.directory/'pressure.png').write_bytes(b'corrupt')
        self.assertEqual(checker.check_run(self.directory)['state'],'failed')
        self.assertFalse((self.directory/'pressure.png').exists())

    def test_cli_exit_codes_for_scheduler(self):
        command=[sys.executable,str(Path(checker.__file__).resolve()),'--output',str(self.directory)]
        self.status(state='starting')
        running=subprocess.run(command,text=True,capture_output=True)
        self.assertEqual(running.returncode,2,running.stderr)
        self.assertEqual(json.loads(running.stdout)['state'],'running')
        self.status(state='failed',error='known failure')
        failed=subprocess.run(command,text=True,capture_output=True)
        self.assertEqual(failed.returncode,1,failed.stderr)
        self.assertEqual(json.loads(failed.stdout)['state'],'failed')
        (self.directory/'completion.json').unlink()
        self.status(state='complete');self.results()
        complete=subprocess.run(command,text=True,capture_output=True)
        self.assertEqual(complete.returncode,0,complete.stderr)
        self.assertEqual(json.loads(complete.stdout)['state'],'complete')
