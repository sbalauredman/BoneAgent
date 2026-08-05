from __future__ import annotations

import logging
from pathlib import Path

from boneagent.console.common import configure_logging, parser
from boneagent.data.tabular import write_manifest

logger = logging.getLogger(__name__)


def main() -> None:
    argument_parser = parser("Build a verified data manifest")
    argument_parser.add_argument("--input", required=True, type=Path, nargs="+")
    argument_parser.add_argument("--output", required=True, type=Path)
    arguments = argument_parser.parse_args()
    configure_logging(arguments.verbose)
    write_manifest(arguments.input, arguments.output)
    logger.info("manifest written with %d files", len(arguments.input))


if __name__ == "__main__":
    main()
