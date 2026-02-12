import logging
import os
import shutil
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    run_id: str
    accuracy: float
    loss: float
    params: dict


class MetricsPublisher:
    def __init__(self, pushgateway_url: str):
        self.pushgateway_url = pushgateway_url
        self.registry = CollectorRegistry()
        self.accuracy_gauge = Gauge(
            "ml_training_accuracy",
            "Accuracy of ML training run",
            ["run_id", "experiment_name", "C", "max_iter"],
            registry=self.registry,
        )
        self.loss_gauge = Gauge(
            "ml_training_loss",
            "Log loss of ML training run",
            ["run_id", "experiment_name", "C", "max_iter"],
            registry=self.registry,
        )

    def push(self, result: RunResult, experiment_name: str):
        labels = [
            result.run_id,
            experiment_name,
            str(result.params["C"]),
            str(result.params["max_iter"]),
        ]
        self.accuracy_gauge.labels(*labels).set(result.accuracy)
        self.loss_gauge.labels(*labels).set(result.loss)

        try:
            push_to_gateway(
                self.pushgateway_url,
                job="ml_training",
                grouping_key={"run_id": result.run_id},
                registry=self.registry,
            )
            logger.info("Метрики надіслано до PushGateway для run_id=%s", result.run_id)
        except Exception as e:
            logger.error("Помилка надсилання до PushGateway: %s", e)


class ExperimentRunner:
    HYPERPARAMS = [
        {"C": 0.01, "max_iter": 50},
        {"C": 0.1, "max_iter": 100},
        {"C": 1.0, "max_iter": 100},
        {"C": 10.0, "max_iter": 200},
        {"C": 0.1, "max_iter": 200},
        {"C": 1.0, "max_iter": 50},
    ]

    def __init__(
        self,
        tracking_uri: str,
        pushgateway_url: str,
        experiment_name: str = "Iris Classification",
        best_model_dir: str = "best_model",
    ):
        self.experiment_name = experiment_name
        self.best_model_dir = best_model_dir
        self.publisher = MetricsPublisher(pushgateway_url)

        mlflow.set_tracking_uri(tracking_uri)
        self.experiment_id = self._get_or_create_experiment()

    def _get_or_create_experiment(self) -> str:
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(self.experiment_name)
            logger.info("Створено експеримент '%s' (ID=%s)", self.experiment_name, experiment_id)
            return experiment_id
        logger.info("Використовується існуючий експеримент '%s' (ID=%s)", self.experiment_name, experiment.experiment_id)
        return experiment.experiment_id

    def _train_single_run(self, params: dict, X_train, X_test, y_train, y_test) -> RunResult:
        with mlflow.start_run(experiment_id=self.experiment_id) as run:
            run_id = run.info.run_id
            logger.info("Запуск %s | C=%s, max_iter=%s", run_id, params["C"], params["max_iter"])

            mlflow.log_params(params)

            model = LogisticRegression(C=params["C"], max_iter=params["max_iter"], random_state=42)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            acc = accuracy_score(y_test, y_pred)
            loss = log_loss(y_test, y_proba)

            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("loss", loss)
            mlflow.sklearn.log_model(model, "model")

            logger.info("accuracy=%.4f, loss=%.4f", acc, loss)

            return RunResult(run_id=run_id, accuracy=acc, loss=loss, params=params)

    def _save_best_model(self, best: RunResult):
        logger.info("Найкращий запуск: %s (accuracy=%.4f)", best.run_id, best.accuracy)

        if os.path.exists(self.best_model_dir):
            shutil.rmtree(self.best_model_dir)

        artifact_uri = f"runs:/{best.run_id}/model"
        mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri, dst_path=self.best_model_dir)
        logger.info("Модель збережено у '%s/'", self.best_model_dir)

    def run(self):
        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        results: list[RunResult] = []

        for params in self.HYPERPARAMS:
            result = self._train_single_run(params, X_train, X_test, y_train, y_test)
            self.publisher.push(result, self.experiment_name)
            results.append(result)

        best = max(results, key=lambda r: r.accuracy)
        self._save_best_model(best)
        logger.info("Готово! Виконано %d запусків.", len(results))


def main():
    runner = ExperimentRunner(
        tracking_uri=os.environ["MLFLOW_TRACKING_URI"],
        pushgateway_url=os.environ.get("PUSHGATEWAY_URL", "http://localhost:9091"),
        best_model_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "best_model"),
    )
    runner.run()


if __name__ == "__main__":
    main()