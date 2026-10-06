# Local Development Guide

## 1. Get the repository

The safest option on a personal computer is to clone the public repository directly:

```bash
git clone https://github.com/kokeik/recsys-challenge-2026.git
cd recsys-challenge-2026
```

Cloning preserves Git history and avoids copying credentials from the laboratory server.

If the project folder is downloaded from the server instead, make sure hidden files are
included. Verify that `.git/` exists and that the remote is correct:

```bash
git status
git remote -v
```

If `.git/` was omitted by the download tool, prefer cloning from GitHub and copy only the
new uncommitted files into the clone.

## 2. Set up Python

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install all development and modeling dependencies:

```bash
python -m pip install --upgrade pip
pip install -e ".[all]"
```

## 3. Verify the checkout

```bash
pytest -q
ruff check src tests
python -m turnaware_recsys.demo
```

Expected unit-test baseline: four passing tests. The synthetic demo should print its
backend, turn-wise metrics, and example Top-5 rankings.

## 4. Use Codex locally

Open the `recsys-challenge-2026` directory as the workspace. Codex will discover the
root-level `AGENTS.md` automatically. A useful first request is:

```text
Read AGENTS.md, README.md, and reports/experiment_summary.md. Run the existing tests,
then explain the architecture and propose the smallest high-value portfolio improvement.
```

For implementation requests, ask Codex to run tests and show the resulting Git diff. Keep
one conceptual change per commit.

## 5. Authenticate only on the personal computer

Use either your normal GitHub SSH key or GitHub CLI on the personal computer. Do not copy
the laboratory server's temporary Deploy Keys; those keys were deleted after use.

With GitHub CLI:

```bash
gh auth login
git push
```

With an existing SSH key, change the remote if desired:

```bash
git remote set-url origin git@github.com:kokeik/recsys-challenge-2026.git
git push
```

## 6. Before publishing changes

Check all of the following:

- Tests and lint pass.
- No dataset rows, blind-set files, embeddings, model weights, or credentials are staged.
- Benchmark values are labeled as development-set observations from the team environment.
- Synthetic-demo metrics are not presented as competition results.
- Claims distinguish Konishi's contribution from the full team submission.

Useful commands:

```bash
git status --short
git diff --check
git diff --cached
```

