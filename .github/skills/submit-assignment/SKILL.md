---
name: submit-assignment
description: Guarded scripted submission when the user says submit assignment 17.
---

# Submit assignment 17

Do not edit files during submission. Run from the repository root:

`python .github/skills/submit-assignment/scripts/submit.py --assignment assignment17 --dry-run`

Report the plan and wait for explicit execution approval. Then run:

`python .github/skills/submit-assignment/scripts/submit.py --assignment assignment17 --execute`

On a nonzero exit, stop and report the reason. Do not replace the script with manual
Git commands or bypass its checks. A push failure can leave a local commit; report
that and rerun only after the user resolves the issue. Report the verified commit,
worktree state and GitHub Actions result separately. Pending or unavailable checks
are not success. The protected submission-policy.json owns tests and exact paths.
