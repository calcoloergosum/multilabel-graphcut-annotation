from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import cv2
from multilabel_graphcut.app import CACHE_DIR, event_loop
from multilabel_graphcut.common import MultiLabelState
from multilabel_graphcut.labels import read_labels


def main():
    parser = ArgumentParser()
    parser.add_argument("filename", type=Path)
    parser.add_argument("save_to", type=Path)
    parser.add_argument("--label-config", "-c", type=Path, required=True)
    args = parser.parse_args()

    labels = read_labels(args.label_config)
    state = MultiLabelState.load(CACHE_DIR)
    state = event_loop(state, labels)

    save_to = args.save_to

    save_label = save_to / f"{args.filename}_label.png"
    save_image = save_to / f"{args.filename}_image.png"
    save_labelmask = save_to / f"{args.filename}_labelmask.toml"
    cv2.imwrite(save_label.as_posix(), state.labelmap)
    cv2.imwrite(save_image.as_posix(), state.image)
    import code
    code.interact(local={**globals(), **locals()})


if __name__ == '__main__':
    main()
