from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from boneagent.analysis.metrics import regression_metrics
from boneagent.analysis.statistics import percentile_bootstrap
from boneagent.console.common import atomic_json, configure_logging, parser

logger = logging.getLogger(__name__)


def main() -> None:
    argument_parser = parser("Evaluate BoneAgent property predictions")
    argument_parser.add_argument("--target", required=True, type=Path)
    argument_parser.add_argument("--prediction", required=True, type=Path)
    argument_parser.add_argument("--output", required=True, type=Path)
    arguments = argument_parser.parse_args()
    configure_logging(arguments.verbose)
    target = np.loadtxt(arguments.target, delimiter=",")
    prediction = np.loadtxt(arguments.prediction, delimiter=",")
    values = regression_metrics(target, prediction)
    absolute_errors = np.abs(target - prediction)
    interval = percentile_bootstrap(absolute_errors, resamples=10000)
    atomic_json(
        arguments.output,
        {
            "mean_absolute_error": values.mean_absolute_error,
            "root_mean_squared_error": values.root_mean_squared_error,
            "coefficient_of_determination": values.coefficient_of_determination,
            "spearman_correlation": values.spearman_correlation,
            "mae_confidence_interval": {
                "estimate": interval.estimate,
                "lower": interval.lower,
                "upper": interval.upper,
                "confidence": interval.confidence,
            },
        },
    )
    logger.info("evaluation completed")


if __name__ == "__main__":
    main()
