"""Training job tracker and history management for neuroevolution experiments."""

import os
import sys
import json
import uuid
import shutil
from datetime import datetime
from typing import Any


class OutputTee:
    """Context manager to tee stdout and stderr to both the console and a log file."""

    class _TeeStream:
        def __init__(self, original, file):
            self.original = original
            self.file = file

        def write(self, data):
            self.original.write(data)
            self.original.flush()
            try:
                self.file.write(data)
                self.file.flush()
            except Exception:
                pass

        def flush(self):
            self.original.flush()
            try:
                self.file.flush()
            except Exception:
                pass

        def fileno(self):
            return self.original.fileno()

        def isatty(self):
            return getattr(self.original, "isatty", lambda: False)()

    def __init__(self, log_path: str):
        self.log_path = log_path
        self.file = None
        self._orig_stdout = None
        self._orig_stderr = None

    def __enter__(self):
        self.file = open(self.log_path, "a", encoding="utf-8")
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = self._TeeStream(self._orig_stdout, self.file)
        sys.stderr = self._TeeStream(self._orig_stderr, self.file)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._orig_stdout:
            sys.stdout = self._orig_stdout
        if self._orig_stderr:
            sys.stderr = self._orig_stderr
        if self.file:
            try:
                self.file.close()
            except Exception:
                pass


class TrainingJob:
    """Represents an individual training job with its dedicated directory, artifacts, and metrics."""

    def __init__(self, job_dir: str, job_id: str, config: dict[str, Any] | None = None):
        self.job_dir = job_dir
        self.job_id = job_id
        self.config = config or {}
        os.makedirs(self.job_dir, exist_ok=True)
        if self.config:
            self.save_config(self.config)

    def save_config(self, config: dict[str, Any]) -> str:
        """Save configuration parameters to config.json."""
        self.config.update(config)
        config_path = os.path.join(self.job_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2, default=str)
        return config_path

    def save_metrics(self, metrics: dict[str, Any]) -> str:
        """Save training metrics and evaluation summary to metrics.json."""
        metrics_path = os.path.join(self.job_dir, "metrics.json")
        existing_metrics = {}
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, "r", encoding="utf-8") as f:
                    existing_metrics = json.load(f)
            except Exception:
                existing_metrics = {}
        existing_metrics.update(metrics)
        if "completed_at" not in existing_metrics:
            existing_metrics["completed_at"] = datetime.now().isoformat()
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(existing_metrics, f, indent=2, default=str)
        return metrics_path

    def capture_output(self, filename: str = "train.log") -> OutputTee:
        """Context manager to tee all stdout and stderr into job_dir/train.log."""
        log_path = os.path.join(self.job_dir, filename)
        return OutputTee(log_path)

    def log(self, message: str) -> None:
        """Append a log line to train.log."""
        log_path = os.path.join(self.job_dir, "train.log")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {message}\n")

    def promote(
        self,
        checkpoint_dir: str = "checkpoints",
        assets_dir: str = "assets",
        model_prefix: str = "model",
    ) -> list[str]:
        """Copy the best model, weights, plot, and gif to top-level checkpoints and assets directories."""
        promoted_files = []
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(assets_dir, exist_ok=True)

        # 1. PyTorch model
        src_pth = os.path.join(self.job_dir, f"{model_prefix}_best.pth")
        if os.path.exists(src_pth):
            dst_pth = os.path.join(checkpoint_dir, f"{model_prefix}_best.pth")
            shutil.copy2(src_pth, dst_pth)
            promoted_files.append(dst_pth)

        # 2. Weights numpy array
        src_npy = os.path.join(self.job_dir, f"{model_prefix}_weights.npy")
        if os.path.exists(src_npy):
            dst_npy = os.path.join(checkpoint_dir, f"{model_prefix}_weights.npy")
            shutil.copy2(src_npy, dst_npy)
            promoted_files.append(dst_npy)

        # 3. Learning curve plot
        src_plot = os.path.join(self.job_dir, f"{model_prefix}_learning_curve.png")
        if os.path.exists(src_plot):
            dst_plot = os.path.join(checkpoint_dir, f"{model_prefix}_learning_curve.png")
            shutil.copy2(src_plot, dst_plot)
            promoted_files.append(dst_plot)

        # 4. Recorded GIF demo
        src_gif = os.path.join(self.job_dir, f"{model_prefix}.gif")
        if os.path.exists(src_gif):
            dst_gif = os.path.join(assets_dir, f"{model_prefix}.gif")
            shutil.copy2(src_gif, dst_gif)
            promoted_files.append(dst_gif)

        return promoted_files


def create_job(
    env_name: str,
    config: dict[str, Any] | None = None,
    base_dir: str = "jobs",
    job_id: str | None = None,
) -> TrainingJob:
    """Create a new TrainingJob instance with a unique directory jobs/train-{env}-{uuid}."""
    clean_env = env_name.lower().replace("-", "_").replace(" ", "_")
    if job_id is None:
        short_id = uuid.uuid4().hex[:8]
        job_id = f"train-{clean_env}-{short_id}"

    job_dir = os.path.join(base_dir, job_id)
    initial_config = {
        "job_id": job_id,
        "env_name": env_name,
        "created_at": datetime.now().isoformat(),
    }
    if config:
        initial_config.update(config)

    return TrainingJob(job_dir=job_dir, job_id=job_id, config=initial_config)


def list_jobs(base_dir: str = "jobs") -> list[dict[str, Any]]:
    """Scan base_dir and return a list of summaries for all stored training jobs."""
    if not os.path.isdir(base_dir):
        return []

    jobs = []
    for entry in sorted(os.listdir(base_dir)):
        job_path = os.path.join(base_dir, entry)
        if not os.path.isdir(job_path):
            continue

        config_file = os.path.join(job_path, "config.json")
        metrics_file = os.path.join(job_path, "metrics.json")

        config = {}
        if os.path.exists(config_file):
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    config = json.load(f)
            except Exception:
                pass

        metrics = {}
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, "r", encoding="utf-8") as f:
                    metrics = json.load(f)
            except Exception:
                pass

        summary = {
            "job_id": entry,
            "job_dir": job_path,
            "env_name": config.get("env_name", "Unknown"),
            "created_at": config.get("created_at", "Unknown"),
            "generations": config.get("generations", config.get("n_generations", "-")),
            "population": config.get("population", config.get("population_size", "-")),
            "crossover_prob": config.get("crossover_prob", "-"),
            "mutation_prob": config.get("mutation_prob", "-"),
            "hidden_layers": config.get("hidden_layers", "-"),
            "n_workers": config.get("n_workers", "-"),
            "best_fitness": metrics.get("best_fitness", "-"),
            "avg_eval_reward": metrics.get("avg_eval_reward", "-"),
            "has_model": os.path.exists(os.path.join(job_path, f"{config.get('model_prefix', 'model')}_best.pth")),
            "has_plot": os.path.exists(os.path.join(job_path, f"{config.get('model_prefix', 'model')}_learning_curve.png")),
        }
        jobs.append(summary)

    # Sort by creation date descending
    jobs.sort(key=lambda x: str(x.get("created_at", "")), reverse=True)
    return jobs
