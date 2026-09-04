"""Tests for training job tracker and history management."""

import os
import json
import shutil
import tempfile
import unittest
import numpy as np

from neuroevolution.tracker import TrainingJob, create_job, list_jobs
from neuroevolution.cli import print_jobs_table


class TestTracker(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_create_job(self):
        config = {"generations": 50, "population": 20, "mutation_prob": 0.15}
        job = create_job("CarRacing-v3", config=config, base_dir=self.test_dir)

        self.assertTrue(os.path.isdir(job.job_dir))
        self.assertTrue(job.job_id.startswith("train-carracing_v3-"))
        self.assertTrue(os.path.exists(os.path.join(job.job_dir, "config.json")))

        with open(os.path.join(job.job_dir, "config.json"), "r") as f:
            saved = json.load(f)
        self.assertEqual(saved["env_name"], "CarRacing-v3")
        self.assertEqual(saved["generations"], 50)
        self.assertEqual(saved["population"], 20)

    def test_job_logging_and_metrics(self):
        job = create_job("CartPole", base_dir=self.test_dir)
        job.log("Starting training run")
        job.log("Generation 1 complete: best=25.0")

        log_path = os.path.join(job.job_dir, "train.log")
        self.assertTrue(os.path.exists(log_path))
        with open(log_path, "r") as f:
            log_content = f.read()
        self.assertIn("Starting training run", log_content)
        self.assertIn("Generation 1 complete", log_content)

        job.save_metrics({
            "best_fitness": 480.0,
            "success": True,
            "generations_completed": 15,
        })
        metrics_path = os.path.join(job.job_dir, "metrics.json")
        self.assertTrue(os.path.exists(metrics_path))
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        self.assertEqual(metrics["best_fitness"], 480.0)
        self.assertTrue(metrics["success"])
        self.assertIn("completed_at", metrics)

    def test_job_promote(self):
        job = create_job("carracing", base_dir=self.test_dir)
        model_prefix = "carracing"

        # Create dummy artifacts inside job_dir
        with open(os.path.join(job.job_dir, f"{model_prefix}_best.pth"), "w") as f:
            f.write("dummy_model_bytes")
        np.save(os.path.join(job.job_dir, f"{model_prefix}_weights.npy"), np.array([1.0, 2.0, 3.0]))
        with open(os.path.join(job.job_dir, f"{model_prefix}_learning_curve.png"), "w") as f:
            f.write("dummy_png")
        with open(os.path.join(job.job_dir, f"{model_prefix}.gif"), "w") as f:
            f.write("dummy_gif")

        ckpt_dir = os.path.join(self.test_dir, "checkpoints")
        assets_dir = os.path.join(self.test_dir, "assets")

        promoted = job.promote(checkpoint_dir=ckpt_dir, assets_dir=assets_dir, model_prefix=model_prefix)
        self.assertEqual(len(promoted), 4)

        self.assertTrue(os.path.exists(os.path.join(ckpt_dir, f"{model_prefix}_best.pth")))
        self.assertTrue(os.path.exists(os.path.join(ckpt_dir, f"{model_prefix}_weights.npy")))
        self.assertTrue(os.path.exists(os.path.join(ckpt_dir, f"{model_prefix}_learning_curve.png")))
        self.assertTrue(os.path.exists(os.path.join(assets_dir, f"{model_prefix}.gif")))

    def test_list_jobs(self):
        # Empty list when dir doesn't exist
        self.assertEqual(list_jobs(os.path.join(self.test_dir, "nonexistent")), [])

        # Create two jobs
        job1 = create_job("CartPole", config={"generations": 10}, base_dir=self.test_dir)
        job1.save_metrics({"best_fitness": 50.0})

        job2 = create_job("LunarLander", config={"generations": 100}, base_dir=self.test_dir)
        job2.save_metrics({"best_fitness": 220.0})

        jobs = list_jobs(base_dir=self.test_dir)
        self.assertEqual(len(jobs), 2)
        env_names = {j["env_name"] for j in jobs}
        self.assertEqual(env_names, {"CartPole", "LunarLander"})

        # Test table printer works cleanly
        print_jobs_table(jobs)
        print_jobs_table([])


if __name__ == "__main__":
    unittest.main()
