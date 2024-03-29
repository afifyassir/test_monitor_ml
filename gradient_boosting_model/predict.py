import logging
import typing as t

import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gradient_boosting_model import __version__ as _version  # noqa: E402
from gradient_boosting_model.config.core import config  # noqa: E402
from gradient_boosting_model.processing.data_management import load_pipeline  # noqa: E402
from gradient_boosting_model.processing.validation import validate_inputs  # noqa: E402

_logger = logging.getLogger(__name__)

pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
_price_pipe = load_pipeline(file_name=pipeline_file_name)


def make_prediction(*, input_data: t.Union[pd.DataFrame, dict],) -> dict:
    """Make a prediction using a saved model pipeline."""

    data = pd.DataFrame(input_data)
    validated_data, errors = validate_inputs(input_data=data)
    results = {"predictions": None, "version": _version, "errors": errors}

    if not errors:
        predictions = _price_pipe.predict(
            X=validated_data[config.model_config.features]
        )
        _logger.info(
            f"Making predictions with model version: {_version} "
            f"Predictions: {predictions}"
        )
        results = {"predictions": predictions, "version": _version, "errors": errors}

    return results
