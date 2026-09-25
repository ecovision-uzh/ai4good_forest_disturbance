"""Load a YAML config and apply command-line overrides.

    config = load_config("configs/statistical_mlp.yaml", ["data.fold=1", "trainer.max_epochs=3"])
    config["data"]["fold"]  # -> 1

* Overrides use dots for nesting; values are parsed as YAML (`3` -> int, `[s1,s2]` -> list).
* `${NAME}` inside a string is replaced by the environment variable NAME.
"""

import os
import re
from pathlib import Path

import yaml


def load_config(path: str | Path, overrides: list[str] = ()) -> dict:
    config = yaml.safe_load(Path(path).read_text())
    for override in overrides:
        key, sep, value = override.partition("=")
        if not sep:
            raise ValueError(f"Override {override!r} must look like key.subkey=value.")
        *parents, leaf = key.split(".")
        node = config
        for part in parents:
            if not isinstance(node.get(part), dict):
                raise KeyError(f"Override {override!r}: {part!r} is not a section of the config.")
            node = node[part]
        if leaf not in node:
            raise KeyError(
                f"Override {override!r}: unknown key {leaf!r}. Existing keys: {sorted(node)}."
            )
        node[leaf] = yaml.safe_load(value)
    return _expand_env(config)


def _expand_env(node):
    if isinstance(node, dict):
        return {key: _expand_env(value) for key, value in node.items()}
    if isinstance(node, list):
        return [_expand_env(value) for value in node]
    if isinstance(node, str):
        expanded = os.path.expandvars(node)
        missing = re.findall(r"\$\{(\w+)\}", expanded)
        if missing:
            raise KeyError(
                f"Environment variable {missing[0]} is not set (used in config value {node!r})."
            )
        return expanded
    return node


def save_config(config: dict, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(config, sort_keys=False))
