"""Label definitions"""
import json
from pathlib import Path


def _get_def(path: Path):
    with path.open('r') as f:
        contents = json.load(f)
        for v in contents.values():
            v['color'] = tuple(v['color'])
        return contents


def read_labels(path: Path):
    return [(k, v["key"], v["color"]) for k, v in _get_def(path).items()]
