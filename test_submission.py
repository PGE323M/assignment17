"""Submission checks use temporary repositories; no GitHub sign-in is required.

SUBMISSION_VALIDATION is an inner-test recursion guard, not a security boundary.
Always unset it for the outer/full validation. The numerical tests never use it.
"""
import os
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
SCRIPT = '.github/skills/submit-assignment/scripts/submit.py'
ALLOWED = ['assignment17.py']


@unittest.skipIf(os.environ.get('SUBMISSION_VALIDATION') == '1', 'inner submission validation: avoid harness recursion')
class TestSubmission(unittest.TestCase):
    def setUp(self):
        self.assertTrue((ROOT / SCRIPT).is_file(), 'Create the submission script')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name) / 'repo'
        self.remote = Path(self.temp.name) / 'remote.git'
        self.repo.mkdir()
        self.env = dict(os.environ)
        self.env.pop('SUBMISSION_VALIDATION', None)
        for key in tuple(self.env):
            if key.startswith('GIT_'):
                self.env.pop(key)
        self.env.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull)
        self.call('git', 'init', '--bare', str(self.remote))
        self.call('git', 'init', '-b', 'master')
        self.call('git', 'config', 'user.name', 'Submission Test')
        self.call('git', 'config', 'user.email', 'test@example.invalid')
        for name in ALLOWED + ['submission-policy.json', 'images/grid.png',
                               '.github/skills/submit-assignment/SKILL.md', SCRIPT]:
            dest = self.repo / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, dest)
        (self.repo / '.gitignore').write_text('__pycache__/\n')
        (self.repo / 'test.py').write_text('import unittest\nclass Check(unittest.TestCase):\n def test_ok(self): self.assertTrue(True)\n')
        (self.repo / 'test_completion.py').write_text('import unittest\n')
        (self.repo / 'test_submission.py').write_text('import unittest\n')
        (self.repo / 'protected.txt').write_text('protected\n')
        self.call('git', 'add', '.')
        self.call('git', 'commit', '-m', 'baseline')
        self.call('git', 'remote', 'add', 'origin', str(self.remote))
        self.call('git', 'push', 'origin', 'master')
        self.base = self.call('git', 'rev-parse', 'HEAD').stdout.strip()
        self.change()

    def call(self, *args, check=True):
        return subprocess.run(args, cwd=self.repo, env=self.env, text=True,
                              capture_output=True, check=check)

    def change(self):
        for name in ALLOWED:
            with (self.repo / name).open('a') as f:
                f.write('\n# reviewed change\n')

    def submit(self, *args):
        return self.call(sys.executable, SCRIPT, '--assignment', 'assignment17', *args, check=False)

    def snapshot(self):
        return tuple(self.call('git', *args).stdout for args in [
            ('rev-parse', 'HEAD'), ('diff', '--cached', '--binary'),
            ('ls-remote', 'origin', 'refs/heads/master')])

    def rejected_unchanged(self, *args):
        before = self.snapshot()
        result = self.submit(*args)
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(before, self.snapshot(), result.stdout + result.stderr)

    def test_default_and_explicit_dry_run(self):
        for args in [(), ('--dry-run',)]:
            before = self.snapshot()
            result = self.submit(*args)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('DRY RUN', result.stdout)
            self.assertEqual(before, self.snapshot())

    def test_unexpected_untracked(self):
        (self.repo / 'surprise.txt').write_text('unexpected\n')
        self.rejected_unchanged('--execute')

    def test_protected_staged_and_unstaged(self):
        for staged in [False, True]:
            (self.repo / 'protected.txt').write_text('changed\n')
            if staged:
                self.call('git', 'add', 'protected.txt')
            self.rejected_unchanged('--execute')

    def test_rename_protected_into_allowed(self):
        self.call('git', 'mv', '-f', 'protected.txt', 'assignment17.py')
        self.rejected_unchanged('--execute')

    def test_deleted_and_symlink_deliverables(self):
        path = self.repo / 'assignment17.py'
        path.unlink()
        self.rejected_unchanged('--execute')
        path.symlink_to('protected.txt')
        self.rejected_unchanged('--execute')

    def test_whitespace_staged_and_unstaged(self):
        with (self.repo / 'assignment17.py').open('a') as f:
            f.write('# trailing whitespace   \n')
        self.rejected_unchanged('--execute')
        self.call('git', 'add', 'images/grid.png')
        self.rejected_unchanged('--execute')

    def test_failing_tests_before_staging(self):
        (self.repo / 'test.py').write_text('import unittest\nclass Check(unittest.TestCase):\n def test_fail(self): self.fail("intentional failure")\n')
        self.call('git', 'add', 'test.py')
        self.call('git', 'commit', '-m', 'fixture failing test')
        self.rejected_unchanged('--execute')

    def test_exact_commit_push_and_idempotence(self):
        result = self.submit('--execute')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('GitHub Actions: unavailable', result.stdout)
        self.assertEqual(self.call('git', 'show', '--format=', '--name-only', 'HEAD').stdout.split(), ALLOWED)
        head = self.call('git', 'rev-parse', 'HEAD').stdout.strip()
        self.assertNotEqual(head, self.base)
        self.assertEqual(self.call('git', 'ls-remote', 'origin', 'refs/heads/master').stdout.split()[0], head)
        self.assertEqual(self.call('git', 'status', '--porcelain').stdout, '')
        again = self.submit('--execute')
        self.assertEqual(again.returncode, 0, again.stdout + again.stderr)
        self.assertEqual(self.call('git', 'rev-parse', 'HEAD').stdout.strip(), head)

    def test_failed_commit_stops_before_push(self):
        hook = self.repo / '.git/hooks/pre-commit'
        hook.write_text('#!/bin/sh\nexit 1\n')
        hook.chmod(0o755)
        result = self.submit('--execute')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.call('git', 'rev-parse', 'HEAD').stdout.strip(), self.base)
        self.assertEqual(self.call('git', 'ls-remote', 'origin', 'refs/heads/master').stdout.split()[0], self.base)

    def test_failed_push_retry_no_extra_commit(self):
        hook = self.remote / 'hooks/pre-receive'
        hook.write_text('#!/bin/sh\nexit 1\n')
        hook.chmod(0o755)
        result = self.submit('--execute')
        self.assertNotEqual(result.returncode, 0)
        head = self.call('git', 'rev-parse', 'HEAD').stdout
        self.assertNotEqual(head.strip(), self.base)
        hook.unlink()
        result = self.submit('--execute')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.call('git', 'rev-parse', 'HEAD').stdout, head)

    def test_protected_policy_cannot_expand_allowlist(self):
        path = self.repo / 'submission-policy.json'
        policy = json.loads(path.read_text())
        policy['allowed'].append('submission-policy.json')
        path.write_text(json.dumps(policy))
        self.rejected_unchanged('--execute')

    def test_unknown_assignment_and_conflicting_modes(self):
        self.rejected_unchanged('--assignment', 'assignment999')
        self.rejected_unchanged('--dry-run', '--execute')

    def test_detached_head(self):
        self.call('git', 'checkout', '--detach')
        self.rejected_unchanged('--execute')


if __name__ == '__main__':
    unittest.main()
