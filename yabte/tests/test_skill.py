"""The packaged agent skill: its examples run, the names it cites exist, and the
installer copies it safely."""

import os
import re
import tempfile
import unittest
from pathlib import Path

import yabte.backtest as ybt
from yabte._skills import SKILL_NAME, install, main, skill_source
from yabte.backtest.runner_cpp import yabte_cpp

SKILL = Path(str(skill_source())) / "SKILL.md"
REQUIRE_CPP = os.environ.get("YABTE_REQUIRE_CPP", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def python_blocks(text: str) -> list[str]:
    return re.findall(r"^```python\n(.*?)^```", text, flags=re.M | re.S)


class SkillContentTestCase(unittest.TestCase):
    def setUp(self):
        self.text = SKILL.read_text(encoding="utf-8")

    def test_frontmatter(self):
        m = re.match(r"---\n(.*?)\n---\n", self.text, flags=re.S)
        self.assertIsNotNone(m, "SKILL.md must start with YAML frontmatter")
        fields = dict(
            line.split(": ", 1) for line in m.group(1).splitlines() if ": " in line
        )
        self.assertEqual(fields.get("name"), SKILL_NAME)
        self.assertTrue(fields.get("description", "").startswith("Use when"))

    def test_examples_run(self):
        """Every ``python`` block runs, in order, in one namespace."""
        if REQUIRE_CPP and yabte_cpp is None:
            self.fail("YABTE_REQUIRE_CPP is set but the C++ extension isn't built")
        blocks = python_blocks(self.text)
        self.assertGreaterEqual(len(blocks), 5)
        namespace = {"__name__": "yabte_skill_example"}
        for i, block in enumerate(blocks):
            with self.subTest(block=i):
                # a "<...>" name: coverage skips it rather than looking for the file
                exec(compile(block, f"<SKILL.md block {i}>", "exec"), namespace)
        # the examples produced trades, and the engine comparison ran if it could
        self.assertGreater(len(namespace["srr"].transaction_history), 0)
        self.assertEqual("srr_cpp" in namespace, yabte_cpp is not None)

    def test_cited_constants_exist(self):
        """Upper-case names in backticks are real enum members or flags."""
        known = {"ADFI_AVAILABLE_AT_OPEN", "ADFI_AVAILABLE_AT_CLOSE", "ADFI_REQUIRED"}
        for enum in (
            ybt.CostType,
            ybt.OrderSizeType,
            ybt.OrderStatus,
            ybt.PositionalOrderCheckType,
        ):
            known |= set(enum.__members__)
        cited = set(re.findall(r"`(?:[A-Za-z]+\.)?([A-Z][A-Z_]{2,})`", self.text))
        self.assertTrue(cited)
        self.assertEqual(cited - known, set(), "SKILL.md cites unknown names")


class InstallSkillTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dest = Path(self.tmp.name) / "skills"

    def tearDown(self):
        self.tmp.cleanup()

    def test_install_then_unchanged(self):
        self.assertEqual(install(self.dest), "installed")
        installed = self.dest / SKILL_NAME / "SKILL.md"
        self.assertEqual(installed.read_bytes(), SKILL.read_bytes())
        self.assertEqual(install(self.dest), "unchanged")

    def test_refuses_to_overwrite_a_different_copy(self):
        install(self.dest)
        installed = self.dest / SKILL_NAME / "SKILL.md"
        installed.write_text("edited", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            install(self.dest)
        self.assertEqual(installed.read_text(encoding="utf-8"), "edited")
        self.assertEqual(install(self.dest, force=True), "updated")
        self.assertEqual(installed.read_bytes(), SKILL.read_bytes())

    def test_extra_file_counts_as_different(self):
        install(self.dest)
        (self.dest / SKILL_NAME / "notes.md").write_text("mine", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            install(self.dest)

    def test_cli(self):
        a, b = self.dest / "a", self.dest / "b"
        self.assertEqual(main(["--dest", str(a), "--dest", str(b)]), 0)
        self.assertTrue((a / SKILL_NAME / "SKILL.md").is_file())
        self.assertTrue((b / SKILL_NAME / "SKILL.md").is_file())
        (a / SKILL_NAME / "SKILL.md").write_text("edited", encoding="utf-8")
        self.assertEqual(main(["--dest", str(a)]), 1)
        self.assertEqual(main(["--dest", str(a), "--force"]), 0)


if __name__ == "__main__":
    unittest.main()
