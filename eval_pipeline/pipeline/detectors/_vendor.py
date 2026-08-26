"""Importing detector repos that were never meant to be importable.

The zoo detectors are published as research repos, not packages: no setup.py,
no pyproject, `import networks` assuming the cwd is the repo root. So they are
cloned by hand under `third_party/` and put on `sys.path` for the duration of
one import. See the "Detector zoo" section of the README for the clone commands.

`third_party/` is gitignored, so nothing third-party is committed and a clone of
this repo cannot reconstruct the zoo on its own. What reproducibility there is
lives in the SHAs recorded below: advisory rather than enforced -- a mismatch
warns and keeps going, because pinning someone else's research repo is a
preference, not a correctness requirement -- but never silent, because "which
commit produced this number" is exactly the question a stale checkout makes
unanswerable.

The path is added and removed around the import rather than left in place: these
repos use short top-level module names (`networks`, `models`, `src`) that would
otherwise shadow each other and anything else on the path.
"""

import contextlib
import os
import subprocess
import sys
import warnings
from pathlib import Path

THIRD_PARTY = Path(os.environ.get("ZOO_THIRD_PARTY", "third_party"))
"""Where the clones live. Relative to the cwd, which the README's usage assumes
is the repo root; override with ZOO_THIRD_PARTY for a shared read-only copy."""

REPOS: dict[str, tuple[str, str]] = {
    # directory name -> (clone url, the SHA this integration was verified against)
    "B-Free": ("https://github.com/grip-unina/B-Free.git", ""),
    "GAPL": ("https://github.com/UltraCapture/GAPL.git", ""),
    "rine": ("https://github.com/mever-team/rine.git", ""),
}


def _check_pin(repo: str, root: Path) -> None:
    """Warn if the checkout is not the commit this adapter was written against."""
    _, pin = REPOS[repo]
    if not pin:
        return
    try:
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return  # not a git checkout: the user's arrangement, not an error
    if not head.startswith(pin):
        warnings.warn(
            f"{repo} is checked out at {head[:12]}, but this adapter was verified "
            f"against {pin[:12]}. Upstream may have changed the API it uses.",
            stacklevel=3,
        )


@contextlib.contextmanager
def vendored(repo: str, subdir: str = ""):
    """Put a vendored repo on sys.path for the duration of an import.

    Use it around the import itself, inside a detector's __init__ -- never at
    module scope. `pipeline.detectors` must stay importable on a machine with no
    third_party/ at all, or the whole test suite needs the zoo cloned to run.
    """
    root = THIRD_PARTY / repo
    if not root.is_dir():
        url, _ = REPOS[repo]
        raise FileNotFoundError(
            f"{root.resolve()} is missing. Clone it with:\n"
            f"    git clone {url} {root}\n"
            "See the 'Detector zoo' section of the README for weights too."
        )
    _check_pin(repo, root)

    path = str((root / subdir).resolve())
    sys.path.insert(0, path)
    try:
        yield root / subdir
    finally:
        with contextlib.suppress(ValueError):
            sys.path.remove(path)


def require_file(path: str | Path, what: str, hint: str) -> Path:
    """Resolve a weights path, or fail with something the user can act on.

    Weights are downloaded by hand and gitignored, so a missing file is the
    single most likely first-run failure. It is worth a sentence saying where to
    get it rather than a bare FileNotFoundError from torch.load.
    """
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"{what} not found at {resolved.resolve()}.\n    {hint}")
    return resolved
