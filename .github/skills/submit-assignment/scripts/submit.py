"""Deterministic submission; configuration comes from the protected policy, not CLI arguments."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def run(*args, check=True, env=None):
    result = subprocess.run(args, text=True, capture_output=True, env=env)
    if check and result.returncode:
        raise RuntimeError(f"{' '.join(args)} failed:\n{result.stdout}{result.stderr}")
    return result


def git(*args):
    return run('git', *args).stdout.strip()


def changed():
    # --no-renames makes both the deleted source and added destination visible.
    names = set()
    for args in [('diff', '--name-only', '--no-renames', '-z'),
                 ('diff', '--cached', '--name-only', '--no-renames', '-z'),
                 ('ls-files', '--others', '--exclude-standard', '-z')]:
        names.update(filter(None, run('git', *args).stdout.split('\0')))
    return names


def submission_changes():
    return changed()


def inspect(allowed):
    unexpected = submission_changes() - set(allowed)
    if unexpected:
        raise RuntimeError('Unexpected/protected changes: ' + ', '.join(sorted(unexpected)))
    for name in allowed:
        p = Path(name)
        if not p.is_file() or p.is_symlink() or any(q.is_symlink() for q in p.parents):
            raise RuntimeError('Missing/deleted/symlinked deliverable: ' + name)
    print('Worktree:', git('status', '--short') or 'clean')


def actions_status(sha):
    try:
        remote = git('remote', 'get-url', 'origin')
        if not (remote.startswith('git@github.com:') or remote.startswith('https://github.com/')):
            return 'unavailable (origin is not GitHub)'
        result = run('gh', 'run', 'list', '--commit', sha, '--json', 'status,conclusion,headSha', check=False)
        if result.returncode:
            return 'unavailable (GitHub CLI/auth/network)'
        runs = [r for r in json.loads(result.stdout) if r['headSha'] == sha]
        if not runs or any(r['status'] != 'completed' for r in runs):
            return 'pending (no run yet or checks still running)'
        return 'success' if all(r['conclusion'] == 'success' for r in runs) else 'failure/non-success'
    except (OSError, ValueError, KeyError, RuntimeError):
        return 'unavailable (could not verify checks)'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assignment', required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--dry-run', action='store_true')
    modes.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    root = Path(git('rev-parse', '--show-toplevel')).resolve()
    if root != Path.cwd().resolve():
        raise RuntimeError('Run from the repository root')
    if 'submission-policy.json' in changed() or Path('submission-policy.json').is_symlink():
        raise RuntimeError('Protected submission policy changed')
    policy = json.loads(Path('submission-policy.json').read_text())
    if args.assignment != policy['assignment']:
        raise RuntimeError('Unknown assignment: ' + args.assignment)
    allowed = policy['allowed']
    branch = git('symbolic-ref', '--quiet', '--short', 'HEAD')
    git('remote', 'get-url', 'origin')
    inspect(allowed)
    run('git', 'diff', '--check', '--', *allowed)
    run('git', 'diff', '--cached', '--check')
    env = dict(os.environ, SUBMISSION_VALIDATION='1', PYTHONDONTWRITEBYTECODE='1')
    result = run(sys.executable, '-m', 'unittest', '-v', *policy['tests'], env=env)
    print(result.stdout + result.stderr)
    inspect(allowed)
    run('git', 'diff', '--check', '--', *allowed)
    run('git', 'diff', '--cached', '--check')
    if not args.execute:
        print('DRY RUN: validated; would stage exact deliverables, commit if needed, push origin', branch)
        return
    run('git', 'add', '--', *allowed)
    inspect(allowed)
    run('git', 'diff', '--cached', '--check')
    staged = set(filter(None, run('git', 'diff', '--cached', '--name-only', '-z').stdout.split('\0')))
    if not staged <= set(allowed):
        raise RuntimeError('Unexpected index content')
    if staged:
        run('git', 'commit', '-m', policy['message'])
        if git('diff', '--cached', '--name-only'):
            raise RuntimeError('Commit did not clear the index')
    sha = git('rev-parse', 'HEAD')
    if submission_changes():
        raise RuntimeError('Worktree changed during commit; inspect before retrying')
    run('git', 'push', 'origin', 'HEAD:refs/heads/' + branch)
    remote = git('ls-remote', '--heads', 'origin', 'refs/heads/' + branch).split()
    if not remote or remote[0] != sha:
        raise RuntimeError('Remote commit verification failed')
    print('Commit:', sha)
    print('Worktree:', git('status', '--short') or 'clean')
    print('GitHub Actions:', actions_status(sha))


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, OSError, ValueError, KeyError) as error:
        print('STOP:', error, file=sys.stderr)
        sys.exit(1)
