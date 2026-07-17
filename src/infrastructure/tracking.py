from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from src.config.schema import ExperimentTrackingConfig

logger = logging.getLogger(__name__)


class ExperimentTracker:
    def __init__(self, config: ExperimentTrackingConfig) -> None:
        self.config = config
        self._run = None
        self._enabled = config.enabled and config.provider != "none"
        self._initialized = False

    def init(self, **kwargs: Any) -> None:
        if not self._enabled:
            return
        try:
            if self.config.provider == "wandb":
                import wandb
                self._run = wandb.init(project=self.config.project, **kwargs)
            elif self.config.provider == "mlflow":
                import mlflow
                mlflow.set_experiment(self.config.project)
                self._run = mlflow.start_run(**kwargs)
            elif self.config.provider == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter
                self._run = SummaryWriter(log_dir=f"logs/tb_{self.config.project}")
                if "config" in kwargs:
                    cfg_text = str(kwargs.pop("config", ""))
                    self._run.add_text("config", cfg_text)
            self._initialized = True
        except ImportError as e:
            logger.warning("Tracking provider %s not available: %s", self.config.provider, e)
            self._enabled = False

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if not self._enabled or not self._initialized:
            return
        try:
            if self.config.provider == "wandb" and self._run:
                self._run.log(metrics, step=step)
            elif self.config.provider == "mlflow" and self._run:
                import mlflow
                mlflow.log_metrics(metrics, step=step or 0)
            elif self.config.provider == "tensorboard" and self._run:
                for k, v in metrics.items():
                    self._run.add_scalar(k, v, step or 0)
        except Exception as e:
            logger.debug("Logging failed: %s", e)

    def log_params(self, params: Dict[str, Any]) -> None:
        if not self._enabled or not self._initialized:
            return
        try:
            if self.config.provider == "wandb" and self._run:
                self._run.config.update(params)
            elif self.config.provider == "mlflow" and self._run:
                import mlflow
                mlflow.log_params(params)
        except Exception as e:
            logger.debug("Log params failed: %s", e)

    def finish(self) -> None:
        if not self._enabled or not self._initialized:
            return
        try:
            if self.config.provider == "wandb" and self._run:
                self._run.finish()
            elif self.config.provider == "mlflow" and self._run:
                import mlflow
                mlflow.end_run()
            elif self.config.provider == "tensorboard" and self._run:
                self._run.close()
        except Exception as e:
            logger.debug("Finish failed: %s", e)
