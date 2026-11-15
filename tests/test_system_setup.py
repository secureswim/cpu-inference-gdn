"""Contract tests for the mandatory, machine-specific setup gate."""
import contextlib
import io
import json
import shutil
import subprocess
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import configure
import system_config as setup


class ConfigurationGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = self.root / "system.json"
        self.patch = patch.object(setup, "CONFIG", self.config)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.data = {"schema": setup.SCHEMA, "identity": setup.identity(),
                     "source_signature": setup.source_signature(), "status": "ready",
                     "python": sys.executable, "bin_dir": str(self.root)}
        for name in ("gdn", "machine", "stream", "ceilings"):
            setup.executable(self.data, name).touch()

    def save(self):
        self.config.write_text(json.dumps(self.data), encoding="utf-8")

    def test_missing_setup_has_actionable_command(self):
        with self.assertRaisesRegex(SystemExit, "configure.py first"):
            setup.require_configuration()

    def test_ready_setup_allows_workflow(self):
        self.save()
        self.assertEqual(setup.require_configuration()["status"], "ready")

    def test_other_machine_is_rejected(self):
        self.data["identity"]["host"] = "another host"
        self.save()
        with self.assertRaisesRegex(SystemExit, "another system"):
            setup.require_configuration()

    def test_changed_sources_require_reconfiguration(self):
        self.data["source_signature"] = "old source"
        self.save()
        with self.assertRaisesRegex(SystemExit, "setup has changed"):
            setup.require_configuration()

    def test_partial_setup_allows_only_builds(self):
        self.data["status"] = "building"
        self.save()
        self.assertEqual(setup.require_configuration(for_build=True)["status"], "building")
        with self.assertRaisesRegex(SystemExit, "not completed"):
            setup.require_configuration()

    def test_failed_setup_never_unlocks_build_or_run(self):
        self.data["status"] = "failed"
        self.save()
        for build in (False, True):
            with self.assertRaises(SystemExit):
                setup.require_configuration(for_build=build)

    def test_deleted_executable_requires_setup(self):
        self.save()
        setup.executable(self.data, "ceilings").unlink()
        with self.assertRaisesRegex(SystemExit, "ceilings executable is missing"):
            setup.require_configuration()

    def test_malformed_configuration_has_actionable_command(self):
        self.config.write_text("{", encoding="utf-8")
        with self.assertRaisesRegex(SystemExit, "configure.py first"):
            setup.require_configuration()


class CMakeGateTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("cmake"), "CMake not installed yet")
    def test_unconfigured_checkout_stops_before_compiler_detection(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "scripts").mkdir()
            shutil.copy(setup.ROOT / "CMakeLists.txt", root)
            shutil.copy(setup.ROOT / "scripts/system_config.py", root / "scripts")
            result = subprocess.run([shutil.which("cmake"), "-S", str(root), "-B", str(root / "build"),
                                     f"-DPython3_EXECUTABLE={sys.executable}"], text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("configure.py first", result.stderr)
            self.assertFalse(list((root / "build").rglob("CMakeCCompiler.cmake")))


class ToolchainTests(unittest.TestCase):
    def test_custom_toolchain_requires_both_compilers(self):
        with patch.dict(configure.os.environ, {"CC": "mycc"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "both CC and CXX"):
                configure.select_compilers()

    def test_explicit_compilers_are_preserved(self):
        with patch.dict(configure.os.environ, {"CC": "mycc", "CXX": "mycxx"}, clear=True):
            self.assertEqual(configure.select_compilers(), ("mycc", "mycxx", []))

    def test_windows_activates_existing_msvc(self):
        with patch.dict(configure.os.environ, {}, clear=True), \
             patch.object(configure.platform, "system", return_value="Windows"), \
             patch.object(configure.shutil, "which", return_value=None), \
             patch.object(configure, "activate_msvc", return_value=True):
            self.assertEqual(configure.select_compilers(), ("cl", "cl", []))

    def test_linux_installs_missing_compiler(self):
        def which(name):
            return f"/usr/bin/{name}" if installed[0] else None
        installed = [False]
        with patch.dict(configure.os.environ, {}, clear=True), \
             patch.object(configure.platform, "system", return_value="Linux"), \
             patch.object(configure.shutil, "which", side_effect=which), \
             patch.object(configure, "install_linux_compiler", side_effect=lambda: installed.__setitem__(0, True)) as install:
            self.assertEqual(configure.select_compilers()[:2], ("/usr/bin/gcc", "/usr/bin/g++"))
            install.assert_called_once()

    def test_check_only_does_not_install_or_unlock(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.object(configure, "CONFIG", root / "system.json"), \
                 patch.object(configure, "select_compilers") as compiler, \
                 patch.object(configure, "run") as install, contextlib.redirect_stdout(io.StringIO()):
                args = configure.argparse.Namespace(check=True)
                configure.configure(args)
                self.assertTrue((root / "system-check.json").exists())
                self.assertFalse((root / "system.json").exists())
                compiler.assert_not_called()
                install.assert_not_called()


if __name__ == "__main__":
    unittest.main()
