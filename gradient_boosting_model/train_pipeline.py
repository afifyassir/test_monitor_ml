from sklearn.model_selection import train_test_split
import os

import sys  # noqa: E402
from pathlib import Path  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from gradient_boosting_model import pipeline  # noqa: E402
from gradient_boosting_model.processing.data_management import load_dataset, save_pipeline  # noqa: E402
from gradient_boosting_model.config.core import config  # noqa: E402
from gradient_boosting_model import __version__ as _version  # noqa: E402

import logging  # noqa: E402

# replace 'directory_path' with the path of the directory containing 'gradient_boosting_model'
directory_path = 'testing-and-monitoring-ml-deployements/packages/gradient_boosting_model/gradient_boosting_model'
absolute_directory_path = os.path.abspath(directory_path)

# add this path to sys.path
sys.path.append(absolute_directory_path)

_logger = logging.getLogger(__name__)


def run_training() -> None:
    """Train the model."""

    # read training data
    data = load_dataset(file_name=config.app_config.training_data_file)

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.model_config.features],  # predictors
        data[config.model_config.target],
        test_size=config.model_config.test_size,
        # we are setting the random seed here
        # for reproducibility
        random_state=config.model_config.random_state,
    )

    pipeline.price_pipe.fit(X_train, y_train)

    _logger.warning(f"saving model version: {_version}")
    save_pipeline(pipeline_to_persist=pipeline.price_pipe)


if __name__ == "__main__":
    run_training()
