import importlib.util
from pathlib import Path
import shutil
import subprocess
import unittest

from src.local_models import TIER_MODELS


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("vllm_serve", ROOT / "docker/vllm/serve.py")
serve = importlib.util.module_from_spec(spec)
spec.loader.exec_module(serve)


class VllmSizingTests(unittest.TestCase):
    def test_launcher_models_match_backend_registry(self):
        for family, tiers in serve.MAIN_MODELS.items():
            for tier, (model, _) in tiers.items():
                with self.subTest(family=family, tier=tier):
                    self.assertEqual(model, TIER_MODELS["vllm"][family][tier]["model"])

    def test_tier_recommendations_cover_worst_case_full_budgets(self):
        embed_budget = serve.EMBED_WEIGHTS_GB + serve.EMBED_KV_GB
        for tier, recommendation in (("low", 32), ("mid", 48)):
            largest_weights = max(models[tier][1] for models in serve.MAIN_MODELS.values())
            total = embed_budget + largest_weights + serve.MAIN_KV_GB + serve.HEADROOM_GB
            self.assertLessEqual(total, recommendation, tier)

    @unittest.skipUnless(shutil.which("bash"), "Bash is required to exercise setup.sh")
    def test_setup_thresholds_and_unified_memory_reserve(self):
        cases = ((24, 0, "low", True), (32, 0, "low", False),
                 (47, 0, "low", False), (48, 0, "mid", False),
                 (59, 1, "low", False), (60, 1, "mid", False))
        for memory, unified, expected, warning in cases:
            with self.subTest(memory=memory, unified=unified):
                result = subprocess.run(
                    ["bash", "-c", f"source ./setup.sh; pick_vllm_tier {memory} {unified}"],
                    cwd=ROOT, check=True, capture_output=True, text=True,
                )
                self.assertEqual(result.stdout.strip(), expected)
                self.assertEqual("recommendation" in result.stderr, warning)


if __name__ == "__main__":
    unittest.main()
