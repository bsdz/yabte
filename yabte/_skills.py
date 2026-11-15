"""Install yabte's agent skill into a project.

The skill ships inside the package as ``yabte/skills/yabte/SKILL.md``. Agents that
support skills look for them in a project's ``.claude/skills/`` or
``.agents/skills/``, so ``yabte-install-skill`` copies it there::

    yabte-install-skill                         # into ./.claude/skills and ./.agents/skills
    yabte-install-skill --dest ~/.claude/skills # anywhere else
"""

import argparse
import filecmp
import shutil
import sys
from importlib.resources import as_file, files
from pathlib import Path

SKILL_NAME = "yabte"

DEFAULT_DESTS = [Path(".claude/skills"), Path(".agents/skills")]


def skill_source():
    """The packaged skill directory (a Traversable)."""
    return files("yabte") / "skills" / SKILL_NAME


def _same_files(a: Path, b: Path) -> bool:
    names = sorted(p.name for p in a.iterdir())
    if names != sorted(p.name for p in b.iterdir()):
        return False
    return all(filecmp.cmp(a / n, b / n, shallow=False) for n in names)


def install(dest: Path, force: bool = False) -> str:
    """Copy the skill into ``dest/yabte``.

    Returns what happened: ``"installed"``, ``"updated"`` or ``"unchanged"``.

    Raises ``FileExistsError`` if a different copy is already there and `force` isn't
    set.
    """
    target = Path(dest).expanduser() / SKILL_NAME
    with as_file(skill_source()) as src:
        if target.exists():
            if _same_files(Path(src), target):
                return "unchanged"
            if not force:
                raise FileExistsError(
                    f"{target} exists and differs from this yabte's skill; "
                    "pass --force to replace it"
                )
            if target.is_symlink():
                target.unlink()  # replace the link, not what it points to
            else:
                shutil.rmtree(target)
            shutil.copytree(src, target)
            return "updated"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, target)
        return "installed"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="yabte-install-skill",
        description="Copy yabte's agent skill (SKILL.md) into skills directories.",
    )
    parser.add_argument(
        "--dest",
        action="append",
        type=Path,
        help="skills directory to install into; may be repeated "
        "(default: ./.claude/skills and ./.agents/skills)",
    )
    parser.add_argument(
        "--force", action="store_true", help="replace an existing, different copy"
    )
    args = parser.parse_args(argv)

    status = 0
    for dest in args.dest or DEFAULT_DESTS:
        try:
            outcome = install(dest, force=args.force)
        except FileExistsError as e:
            print(f"yabte-install-skill: {e}", file=sys.stderr)
            status = 1
        else:
            print(f"{outcome}: {Path(dest).expanduser() / SKILL_NAME}")
    return status


if __name__ == "__main__":
    sys.exit(main())
