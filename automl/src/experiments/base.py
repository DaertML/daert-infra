"""Base AutoML experiment class with real data support."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Callable
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, mean_squared_error, mean_absolute_error, r2_score)


class ProgressLogger:
    """Simple progress logger."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.steps = []
        self.current_step = 0
        self.total_steps = 0

    def log(self, message: str) -> None:
        """Log a message."""
        if self.verbose:
            print(f"  → {message}")
        self.steps.append(message)

    def step(self, name: str) -> 'ProgressLogger':
        """Start a new step."""
        self.current_step += 1
        self.log(f"[{self.current_step}/{self.total_steps}] {name}")
        return self


class AutoMLExperiment(ABC):
    """Base class for AutoML experiments."""

    def __init__(self, experiment_id: str, params: Dict[str, Any],
                 progress_callback: Optional[Callable] = None):
        self.experiment_id = experiment_id
        self.params = params
        self.model = None
        self.metrics = {}
        self.feature_names = []
        self.progress_callback = progress_callback or (lambda x, msg: None)
        self.mlflow_run_id = None

    def _progress(self, percent: float, message: str) -> None:
        """Report progress."""
        if self.progress_callback:
            self.progress_callback(percent, message)
        else:
            print(f"\r  [{percent:3d}%] {message}", end="", flush=True)

    @abstractmethod
    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load and return training data (X, y)."""
        pass

    @abstractmethod
    def build_model(self) -> None:
        """Build and configure the model."""
        pass

    @abstractmethod
    def train(self, X_train, y_train) -> None:
        """Train the model."""
        pass

    def evaluate(self, X_test, y_test) -> Dict[str, float]:
        """Evaluate the model and return metrics."""
        y_pred = self.model.predict(X_test)

        # Auto-detect problem type
        n_classes = len(np.unique(y_test))
        is_regression = n_classes > 20 or isinstance(y_test[0], float)

        if is_regression:
            self.metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred)
            }
        else:
            avg = 'weighted' if n_classes > 2 else 'binary'
            self.metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average=avg, zero_division=0),
                'recall': recall_score(y_test, y_pred, average=avg, zero_division=0),
                'f1': f1_score(y_test, y_pred, average=avg, zero_division=0)
            }

        return self.metrics

    def run(self) -> Dict[str, float]:
        """Run the complete experiment pipeline."""
        steps = [
            ("Loading data", 10),
            ("Analyzing columns", 20),
            ("Preprocessing features", 30),
            ("Building train/test split", 40),
            ("Building model", 50),
            ("Training model", 70),
            ("Evaluating metrics", 90),
            ("Saving to MLflow", 100)
        ]

        for step_name, percent in steps:
            self._progress(percent, step_name)

        self.build_model()
        self._progress(10, "Loading data")
        X, y = self.load_data()
        self.feature_names = getattr(X, 'columns', []) if hasattr(X, 'columns') else []

        self._progress(20, f"Analyzing columns - {X.shape[1]} features detected")

        self._progress(30, "Preprocessing features")
        rs = self.params.get('random_state', 42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.params.get('test_size', 0.2),
            random_state=rs
        )
        self._progress(40, f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples (rs={rs})")

        self._progress(50, f"Building {type(self.model).__name__} model")
        self.train(X_train, y_train)

        self._progress(70, "Training in progress...")
        self._progress(90, "Evaluating on test set")
        metrics = self.evaluate(X_test, y_test)

        self._progress(95, "Logging to MLflow")
        # End any existing run before starting a new one
        try:
            mlflow.end_run()
        except Exception:
            pass
        with mlflow.start_run(run_name=self.experiment_id, nested=True) as run:
            self.mlflow_run_id = run.info.run_id
            mlflow.log_params(self.params)

            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(self.model, "model")

            if self.feature_names:
                mlflow.log_param("n_features", len(self.feature_names))

        self._progress(100, "Experiment completed!")
        return metrics


class CSVExperiment(AutoMLExperiment):
    """Experiment that trains on a CSV file."""

    def __init__(self, experiment_id: str, params: Dict[str, Any],
                 progress_callback: Optional[Callable] = None):
        super().__init__(experiment_id, params, progress_callback)
        self.data_path = params.get('data_path')
        self.target_column = params.get('target_column')
        self._X = None
        self._y = None
        self.data_info = {}

    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load data from CSV."""
        from src.experiments.data_loader import load_data

        if not self.data_path:
            raise ValueError("No data_path specified in parameters")

        X, y, info = load_data(
            csv_path=self.data_path,
            target_column=self.target_column,
            select_features=self.params.get('select_features', False),
            k=self.params.get('n_features', 10)
        )

        self.feature_names = info.get('feature_names', [])
        self.data_info = info

        mlflow.log_params({
            'data_path': self.data_path,
            'target_column': info['target_column'],
            'n_samples': info['n_samples'],
            'n_features': info['n_features']
        })

        return X, y

    def build_model(self) -> None:
        """Build a scikit-learn model based on parameters."""
        from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                                      RandomForestRegressor, GradientBoostingRegressor)
        from sklearn.linear_model import LogisticRegression, LinearRegression
        from sklearn.svm import SVC, SVR
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

        model_type = self.params.get('model_type', 'auto')
        task = self.params.get('task', 'auto')

        # Auto-detect task from data if needed
        if task == 'auto':
            if hasattr(self, '_y') and self._y is not None:
                n_classes = len(np.unique(self._y))
                task = 'regression' if n_classes > 20 or isinstance(self._y[0], float) else 'classification'
            else:
                task = 'classification'

        model_params = self.params.get('model_params', {})

        # Default model selection
        if task == 'classification':
            model_map = {
                'random_forest': RandomForestClassifier,
                'gradient_boosting': GradientBoostingClassifier,
                'logistic_regression': LogisticRegression,
                'svm': SVC,
                'knn': KNeighborsClassifier
            }
            default = RandomForestClassifier
        else:
            model_map = {
                'random_forest': RandomForestRegressor,
                'gradient_boosting': GradientBoostingRegressor,
                'linear_regression': LinearRegression,
                'svr': SVR,
                'knn': KNeighborsRegressor
            }
            default = RandomForestRegressor

        if model_type == 'auto':
            model_class = default
        elif model_type in model_map:
            model_class = model_map[model_type]
        else:
            model_class = default

        # Add default parameters if not specified
        if 'n_estimators' not in model_params:
            model_params['n_estimators'] = 100
        if 'random_state' not in model_params:
            model_params['random_state'] = self.params.get('random_state', 42)

        self.model = model_class(**model_params)
        mlflow.log_param('model_type', model_class.__name__)

    def train(self, X_train, y_train) -> None:
        """Train the model."""
        self.model.fit(X_train, y_train)


class SklearnExperiment(AutoMLExperiment):
    """Scikit-learn based experiment (deprecated, use CSVExperiment)."""

    def __init__(self, experiment_id: str, params: Dict[str, Any]):
        super().__init__(experiment_id, params)
        self.model_type = params.get('model_type', 'random_forest')

    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load sample data for demonstration."""
        from sklearn.datasets import load_iris
        data = load_iris()
        return data.data, data.target

    def build_model(self) -> None:
        """Build a scikit-learn model based on parameters."""
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC

        model_map = {
            'random_forest': RandomForestClassifier,
            'gradient_boosting': GradientBoostingClassifier,
            'logistic_regression': LogisticRegression,
            'svm': SVC
        }

        model_class = model_map.get(self.model_type, RandomForestClassifier)
        self.model = model_class(**self.params.get('model_params', {}))

    def train(self, X_train, y_train) -> None:
        """Train the model."""
        self.model.fit(X_train, y_train)


class XGBoostExperiment(AutoMLExperiment):
    """XGBoost based experiment."""

    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load sample data for demonstration."""
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        return data.data, data.target

    def build_model(self) -> None:
        """Build an XGBoost model."""
        import xgboost as xgb
        self.model = xgb.XGBClassifier(**self.params.get('model_params', {}))

    def train(self, X_train, y_train) -> None:
        """Train the model."""
        self.model.fit(X_train, y_train)


class LightGBMExperiment(AutoMLExperiment):
    """LightGBM based experiment."""

    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load sample data for demonstration."""
        from sklearn.datasets import load_iris
        data = load_iris()
        return data.data, data.target

    def build_model(self) -> None:
        """Build a LightGBM model."""
        import lightgbm as lgb
        self.model = lgb.LGBMClassifier(**self.params.get('model_params', {}))

    def train(self, X_train, y_train) -> None:
        """Train the model."""
        self.model.fit(X_train, y_train)


def create_experiment(experiment_id: str, params: Dict[str, Any]) -> AutoMLExperiment:
    """Factory function to create experiment instances."""
    model_type = params.get('model_type', 'sklearn')
    data_path = params.get('data_path')

    # If data_path is provided, use CSV-based experiment
    if data_path:
        return CSVExperiment(experiment_id, params)

    # Otherwise use demo experiments
    if model_type == 'xgboost':
        return XGBoostExperiment(experiment_id, params)
    elif model_type == 'lightgbm':
        return LightGBMExperiment(experiment_id, params)
    else:
        params['model_type'] = params.get('model_type', 'random_forest')
        return SklearnExperiment(experiment_id, params)
