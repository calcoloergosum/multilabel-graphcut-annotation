from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import random

import cv2
from multilabel_graphcut_annotation.app import event_loop, MultiLabelState, read_label_definitions


def main():
    parser = ArgumentParser()
    parser.add_argument("dir", type=Path)
    parser.add_argument("--filename-pattern", "-g", type=str, default="*.jpg")
    parser.add_argument("--out", "-o", type=Path, default=Path("output"))
    parser.add_argument("--label-config", "-c", type=Path, default=Path(__file__).parent / 'label_definitions' / 'human.json')
    args = parser.parse_args()

    labels = read_label_definitions(args.label_config)

    # event loop
    all_files = sorted(args.dir.rglob(args.filename_pattern))
    random.shuffle(all_files)
    for p in all_files:
        save_dir = args.out / p.relative_to(args.dir).parent
        save_dir.mkdir(exist_ok=True, parents=True)
        if MultiLabelState.exists(save_dir, p.stem):
            continue

        print(f"[*] Annotating {p.as_posix()}")
        image = cv2.imread(p.as_posix())
        assert image is not None

        x, y, w, h = cv2.selectROI("annotation tool", image, showCrosshair=False)
        image = image[y: y + h, x: x + w]
        state = MultiLabelState.new(image)
        state = event_loop(state, labels)

        if state is None:
            p.unlink()
            continue
        state.save(save_dir, p.stem)

    print("Done!")

if __name__ == '__main__':
    main()
