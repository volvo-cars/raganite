#
# Copyright 2026 Volvo Car Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Command-line log levels."""

import logging

_handler: logging.Handler | None = None


def configure_logging(verbosity: int) -> None:
    """Warnings by default. ``-v`` adds info. ``-vv`` logs each caught failure."""
    if verbosity >= 2:
        level = logging.DEBUG
    elif verbosity == 1:
        level = logging.INFO
    else:
        level = logging.WARNING
    global _handler
    if _handler is None:
        _handler = logging.StreamHandler()
        _handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    _handler.setLevel(level)
    for name in ("raganite", "benchmark"):
        logger = logging.getLogger(name)
        logger.setLevel(level)
        if _handler not in logger.handlers:
            logger.addHandler(_handler)
        logger.propagate = False
