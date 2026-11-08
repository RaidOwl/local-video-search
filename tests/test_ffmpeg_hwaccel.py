import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, call, patch

from src.ingest import frame_extractor


def completed(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess([], returncode, stdout=stdout, stderr=stderr)


class HardwareProbeTests(unittest.TestCase):
    def test_compiled_backends_without_devices_use_software(self):
        listed = completed(stdout="Hardware acceleration methods:\ncuda\nvaapi\ndrm\nvulkan\n")
        with patch.object(frame_extractor.subprocess, "run", return_value=listed), \
             patch.object(frame_extractor.os.path, "exists", return_value=False), \
             patch.object(frame_extractor, "_vaapi_devices", return_value=[]), \
             patch.object(frame_extractor, "_device_usable") as probe:
            self.assertEqual(frame_extractor._probe_hwaccel(), ("off", None))
        probe.assert_not_called()

    def test_cuda_requires_a_usable_device(self):
        listed = completed(stdout="Hardware acceleration methods:\ncuda\nvaapi\n")
        with patch.object(frame_extractor.subprocess, "run", return_value=listed), \
             patch.object(frame_extractor.os.path, "exists", return_value=True), \
             patch.object(frame_extractor, "_device_usable", return_value=True) as probe:
            self.assertEqual(frame_extractor._probe_hwaccel(), ("cuda", None))
        probe.assert_called_once_with("cuda=lvs:0")

    def test_vaapi_uses_the_render_node_that_passed_the_probe(self):
        listed = completed(stdout="Hardware acceleration methods:\ncuda\nvaapi\n")
        devices = ["/dev/dri/renderD128", "/dev/dri/renderD129"]
        with patch.object(frame_extractor.subprocess, "run", return_value=listed), \
             patch.object(frame_extractor.os.path, "exists", return_value=True), \
             patch.object(frame_extractor, "_vaapi_devices", return_value=devices), \
             patch.object(frame_extractor, "_device_usable", side_effect=[False, False, True]) as probe:
            choice = frame_extractor._probe_hwaccel()
        self.assertEqual(choice, ("vaapi", devices[1]))
        self.assertEqual(probe.call_args_list, [
            call("cuda=lvs:0"), call("vaapi=lvs:/dev/dri/renderD128"),
            call("vaapi=lvs:/dev/dri/renderD129"),
        ])
        with patch.object(frame_extractor, "_HWACCEL_CACHE", choice):
            self.assertEqual(
                frame_extractor._hwaccel_decode_args("auto"),
                ["-hwaccel", "vaapi", "-hwaccel_output_format", "vaapi",
                 "-vaapi_device", devices[1]],
            )

    def test_probe_errors_are_reported_and_do_not_enable_a_backend(self):
        with patch.object(frame_extractor.subprocess, "run", side_effect=OSError("ffmpeg missing")), \
             patch("builtins.print") as log:
            self.assertEqual(frame_extractor._probe_hwaccel(), ("off", None))
        self.assertIn("ffmpeg missing", str(log.call_args))


class HardwareDecodeFallbackTests(unittest.TestCase):
    def test_single_frame_falls_back_and_subsequent_frames_use_software(self):
        extractor = frame_extractor.FrameExtractor(hwaccel="cuda")
        commands = []

        def run(command, **_kwargs):
            commands.append(command)
            if "-hwaccel" in command:
                return completed(returncode=1, stderr="CUDA device not available")
            Path(command[-1]).touch()
            return completed()

        with tempfile.TemporaryDirectory() as directory, \
             patch.object(frame_extractor.subprocess, "run", side_effect=run), \
             patch("builtins.print") as log:
            first = os.path.join(directory, "first.jpg")
            second = os.path.join(directory, "second.jpg")
            self.assertEqual(extractor.extract_frame_at("video.mp4", 0, first), first)
            self.assertEqual(extractor.extract_frame_at("video.mp4", 1, second), second)

        self.assertEqual(len(commands), 3)
        self.assertIn("-hwaccel", commands[0])
        self.assertNotIn("-hwaccel", commands[1])
        self.assertNotIn("-hwaccel", commands[2])
        self.assertEqual(extractor.resolved_hwaccel(), "off")
        self.assertIn("using software decoding", extractor.hwaccel_warning)
        self.assertIn("CUDA device not available", str(log.call_args_list))

    def test_scene_detection_retries_with_software_and_retains_cuts(self):
        extractor = frame_extractor.FrameExtractor(hwaccel="vaapi")
        outputs = [
            completed(returncode=1, stderr="hardware decode unavailable"),
            completed(stderr="[Parsed_showinfo] pts_time:2.5\n"),
        ]
        with patch.object(frame_extractor, "_vaapi_devices", return_value=["/dev/dri/renderD128"]), \
             patch.object(frame_extractor.subprocess, "run", side_effect=outputs) as run, \
             patch("builtins.print"):
            self.assertEqual(extractor._detect_scene_cuts(Path("video.mp4"), 5), [2.5])
        first, second = [args.args[0] for args in run.call_args_list]
        self.assertIn("-hwaccel", first)
        self.assertNotIn("-hwaccel", second)
        self.assertNotIn("hwdownload", second[second.index("-vf") + 1])
        self.assertEqual(extractor.resolved_hwaccel(), "off")

    def test_failed_software_fallback_is_reported(self):
        extractor = frame_extractor.FrameExtractor(hwaccel="cuda")
        outputs = [
            completed(returncode=1, stderr="no CUDA device"),
            completed(returncode=1, stderr="invalid video"),
        ]
        with patch.object(frame_extractor.subprocess, "run", side_effect=outputs), \
             patch("builtins.print") as log:
            self.assertIsNone(extractor.extract_frame_at("bad.mp4", 0, "/tmp/not-created.jpg"))
        self.assertIn("software fallback failed", extractor.hwaccel_warning)
        self.assertIn("invalid video", str(log.call_args_list))


class HardwareSettingsTests(unittest.IsolatedAsyncioTestCase):
    async def test_settings_reports_the_effective_decoder_and_warning(self):
        from src.api import server

        extractor = Mock()
        extractor.resolved_hwaccel.return_value = "off"
        extractor.hwaccel_warning = "CUDA failed; using software decoding."
        service = Mock(spec=["assert_index_compatible"])
        with patch.object(server, "frame_extractor", extractor), \
             patch.object(server, "embed_service", service), \
             patch.object(server, "VIDEO_SCAN_DIRS", []):
            settings = await server.get_settings_endpoint()
        self.assertEqual(settings["ffmpeg_hwaccel_resolved"], "off")
        self.assertEqual(settings["ffmpeg_hwaccel_warning"], extractor.hwaccel_warning)


class SetupDecoderTests(unittest.TestCase):
    def test_shell_setup_auto_detects_in_the_container_not_on_the_host(self):
        setup = Path(__file__).resolve().parents[1] / "setup.sh"
        real_awk = shutil.which("awk")
        if real_awk is None:
            self.fail("awk is required to exercise setup.sh")

        for system, hardware in (("Linux", "nvidia"), ("Darwin", "apple")):
            for fail_pull in (False, True):
                with self.subTest(system=system, fail_pull=fail_pull), tempfile.TemporaryDirectory() as directory:
                    path = Path(directory)
                    shutil.copy(setup, path / "setup.sh")
                    commands = path / "bin"
                    commands.mkdir()
                    stubs = {
                        "uname": """#!/usr/bin/env bash
echo "$TEST_SYSTEM"
""",
                        "awk": """#!/usr/bin/env bash
if [[ "${2:-}" == "/proc/meminfo" ]]; then echo 33554432; else exec "$REAL_AWK" "$@"; fi
""",
                        "sysctl": """#!/usr/bin/env bash
if [[ "$*" == "-n hw.memsize" ]]; then echo 34359738368; else echo "Apple test"; fi
""",
                        "nvidia-smi": """#!/usr/bin/env bash
case "$1" in
    -L) echo "GPU 0: NVIDIA test GPU" ;;
    --query-gpu=name) echo "NVIDIA test GPU" ;;
    --query-gpu=memory.total) echo 24576 ;;
    --query-gpu=compute_cap) echo 8.6 ;;
esac
""",
                        "nvidia-ctk": "#!/usr/bin/env bash\nexit 0\n",
                        "ffmpeg": "#!/usr/bin/env bash\ntouch \"$FFMPEG_MARKER\"\nexit 1\n",
                        "curl": "#!/usr/bin/env bash\nexit 0\n",
                        "pgrep": "#!/usr/bin/env bash\nexit 1\n",
                        "ollama": "#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" >> \"$OLLAMA_CALLS\"\n",
                        "docker": """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$DOCKER_CALLS"
if [[ "$*" == "compose config --services" ]]; then
    printf '%s\\n' qdrant local-vss-api local-vss-frontend ollama-nvidia
elif [[ "$*" == "compose pull" && "$FAIL_PULL" == 1 ]]; then
    exit 1
fi
exit 0
""",
                    }
                    for name, content in stubs.items():
                        command = commands / name
                        command.write_text(content)
                        command.chmod(0o755)
                    marker = path / "host-ffmpeg-used"
                    docker_calls = path / "docker-calls"
                    ollama_calls = path / "ollama-calls"
                    env = {
                        "HOME": directory,
                        "PATH": f"{commands}{os.pathsep}{os.environ['PATH']}",
                        "TEST_SYSTEM": system,
                        "REAL_AWK": real_awk,
                        "FFMPEG_MARKER": str(marker),
                        "DOCKER_CALLS": str(docker_calls),
                        "OLLAMA_CALLS": str(ollama_calls),
                        "FAIL_PULL": str(int(fail_pull)),
                    }
                    result = subprocess.run(
                        ["bash", str(path / "setup.sh")], env=env,
                        capture_output=True, text=True,
                    )
                    self.assertEqual(result.returncode, 0, f"{result.stdout}\n{result.stderr}")
                    settings = (path / ".env").read_text()
                    self.assertIn(f"HARDWARE_TYPE={hardware}\n", settings)
                    self.assertIn("FFMPEG_HWACCEL=auto\n", settings)
                    self.assertNotIn("\nMODELS_DIR=", settings)
                    self.assertTrue((path / "data" / "models").is_dir())
                    self.assertFalse(marker.exists())
                    build_flag = " --build" if fail_pull else ""
                    self.assertIn(
                        f"compose up -d --remove-orphans{build_flag}",
                        docker_calls.read_text().splitlines(),
                    )
                    if system == "Darwin":
                        self.assertEqual(ollama_calls.read_text().splitlines(), [
                            "pull qwen3-embedding:4b", "pull qwen3-vl:8b-instruct",
                        ])

    def test_shell_setup_preserves_a_user_decoder_choice(self):
        setup = Path(__file__).resolve().parents[1] / "setup.sh"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / ".env").write_text("FFMPEG_HWACCEL=off\nHARDWARE_TYPE=nvidia\n")
            result = subprocess.run(
                ["bash", "-c", 'source "$1"; cd "$2"; configured_ffmpeg_hwaccel',
                 "setup-test", str(setup), directory],
                check=True, capture_output=True, text=True,
            )
        self.assertEqual(result.stdout, "off")


class SetupWizardTests(unittest.TestCase):
    def run_wizard(self, answers, *, curl_response=None, hardware="nvidia", vllm=False):
        setup = Path(__file__).resolve().parents[1] / "setup.sh"
        script = r'''
source "$SETUP_SCRIPT"
cd "$TEST_CWD"
NON_INTERACTIVE=0
USE_VLLM="${TEST_VLLM:-0}"
prompt_value() {
    local value
    read -r value
    printf '%s' "$value"
}
configure_wizard "$TEST_HARDWARE" "NVIDIA test GPU" 24 qwen
wizard_env=""
for setting in "${WIZARD_ENV[@]+"${WIZARD_ENV[@]}"}"; do
    wizard_env+="$setting "
done
printf '\nRESULT|%s|%s|%s|%s|%s\n' "$WIZARD_HW_TYPE" "$WIZARD_TIER" "$WIZARD_FAMILY" "${wizard_env% }" "$WIZARD_MODELS_DIR"
'''
        with tempfile.TemporaryDirectory() as directory:
            env = dict(
                os.environ, SETUP_SCRIPT=str(setup), TEST_CWD=directory,
                LVS_SETUP_TEST_INPUT="1", TEST_HARDWARE=hardware,
                TEST_VLLM=str(int(vllm)), HOME=directory,
            )
            self.home = directory
            if curl_response is not None:
                bin_dir = Path(directory) / "bin"
                bin_dir.mkdir()
                curl = bin_dir / "curl"
                curl.write_text(f"#!/usr/bin/env bash\nprintf '%s' '{curl_response}'\n")
                curl.chmod(0o755)
                env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
            return subprocess.run(
                ["bash", "-c", script, "wizard-test"],
                env=env, input="\n".join(answers) + "\n", capture_output=True, text=True,
            )

    def test_local_wizard_uses_detection_and_allows_tier_and_family_choice(self):
        result = self.run_wizard(["1", "20", "1", "2", "low", "/my videos", "", "Y"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("RESULT|nvidia|low|gemma|VIDEO_DIR_1=/my videos|./data/models\n", result.stdout)
        self.assertIn("Detected system: NVIDIA test GPU", result.stderr)

    def test_wizard_accepts_a_model_storage_folder(self):
        result = self.run_wizard(["1", "20", "1", "1", "mid", "", "~/ai models", "Y"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"RESULT|nvidia|mid|qwen|MODELS_DIR={self.home}/ai models|{self.home}/ai models\n", result.stdout)
        self.assertIn("~13 GB", result.stderr)

    def test_wizard_rejects_a_model_storage_folder_with_a_colon(self):
        result = self.run_wizard(["1", "20", "1", "1", "mid", "", "/a:b"])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("cannot contain ':'", result.stderr)

    def test_vllm_selector_keeps_vllm_as_the_backend_default(self):
        result = self.run_wizard(
            ["1", "24", "2", "1", "low", "", "", "Y"],
            hardware="nvidia-vllm", vllm=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("RESULT|nvidia-vllm|low|qwen|", result.stdout)

    def test_external_wizard_collects_and_verifies_model_endpoints(self):
        response = '{"data":[{"id":"embed-model"},{"id":"vision-model"},{"id":"summary-model"}]}'
        answers = [
            "1", "24", "3",
            "http://embed", "embed-model",
            "http://vision", "vision-model",
            "http://summary", "summary-model",
            "", "", "Y",
        ]
        result = self.run_wizard(answers, curl_response=response)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("RESULT|custom|mid|qwen|", result.stdout)
        for expected in (
            "EMBED_URL=http://embed", "EMBED_MODEL=embed-model",
            "VLM_URL=http://vision", "VLM_MODEL=vision-model",
            "LLM_URL=http://summary", "LLM_MODEL=summary-model",
        ):
            self.assertIn(expected, result.stdout)
        self.assertEqual(result.stderr.count("model verified:"), 3)

    def test_external_wizard_fails_if_model_is_missing(self):
        answers = [
            "1", "24", "3",
            "http://embed", "missing-model",
            "http://vision", "vision-model",
            "http://summary", "summary-model",
        ]
        result = self.run_wizard(answers, curl_response='{"data":[{"id":"other"}]}')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("was not listed", result.stderr)


if __name__ == "__main__":
    unittest.main()
