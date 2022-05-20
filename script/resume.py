from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import cv2
from multilabel_graphcut.app import CACHE_DIR, event_loop
from multilabel_graphcut.common import MultiLabelState
from multilabel_graphcut.labels import read_label_definitions


def main():
    parser = ArgumentParser()
    parser.add_argument("filename", type=Path)
    parser.add_argument("save_to", type=Path)
    parser.add_argument("--label-config", "-c", type=Path, required=True)
    args = parser.parse_args()

    labels = read_label_definitions(args.label_config)
    state = MultiLabelState.load(CACHE_DIR)
    state = event_loop(state, labels)

    state.save(args.save_to, args.filename)


if __name__ == '__main__':
    main()
