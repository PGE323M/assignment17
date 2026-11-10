# Agent instructions

Read README.md, assignment17.py, run_simulation.py, check_simulation.py, test.py,
and inputs.yml before proposing a bounded implementation plan. Wait for student
approval. Edit only the three well methods in assignment17.py. Supplied base
classes, helpers, tests, data, environment, instructions and submission files
are protected. Never weaken tests or silently change governing instructions.
Preserve sparse storage, units, well ordering, repeated-cell addition,
no-well behavior and the course theta convention. Verify the 3x3 case first.
For the agent exercise, follow the README prompt: run the 1000x1000 case once,
use an existing watcher every 30 seconds, produce a pressure contour PNG, then
remove only that watcher. Use the supplied helpers; no new code deliverable or
scheduler implementation. Preserve unrelated tasks and report failures honestly.
Run python -m unittest -v test.py and git diff --check. Ask the student to inspect
the final numerical diff and rerun checks independently. Stop on unexpected
changes, conflicting instructions, failed checks or exhausted memory.
Generated .run/ files stay local. Submit only assignment17.py. No edits during
submission. When asked to submit assignment 17, load
.github/skills/submit-assignment/SKILL.md.
