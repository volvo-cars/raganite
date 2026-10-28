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

"""Explicit configuration loading for command-line applications."""

import json
from pathlib import Path
from typing import Any

import tomllib
from dotenv import load_dotenv
from pydantic_settings import EnvSettingsSource

from .settings import RaganiteConfig


def _merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in updates.items():
        previous = result.get(key)
        if isinstance(previous, dict) and isinstance(value, dict):
            # Switching backends must not carry SQLite-only settings into Neo4j.
            if key == "backend" and "kind" in value and value["kind"] != previous.get("kind"):
                result[key] = dict(value)
            else:
                result[key] = _merge(previous, value)
        else:
            result[key] = value
    return result


def load_config(
    path: Path | None = None,
    *,
    env_file: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> RaganiteConfig:
    """Load settings without searching parent directories for dotenv files.

    Existing process variables take precedence over dotenv values. Environment
    settings then override the config file, and explicit CLI overrides win last.
    Calling this function with env_file=None does not load a dotenv file.
    """
    if env_file is not None:
        if not env_file.is_file():
            raise FileNotFoundError(env_file)
        load_dotenv(env_file, override=False)
    data: dict[str, Any] = {}
    if path is not None:
        if path.suffix.lower() == ".toml":
            with path.open("rb") as stream:
                data = tomllib.load(stream)
        elif path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
        else:
            raise ValueError("Config files must use .json or .toml")
        if not isinstance(data, dict):
            raise ValueError("Config must contain an object of Raganite settings")
    data = _merge(data, EnvSettingsSource(RaganiteConfig)())
    return RaganiteConfig(**_merge(data, overrides or {}))
