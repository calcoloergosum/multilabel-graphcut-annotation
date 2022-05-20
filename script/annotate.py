from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import cv2
from multilabel_graphcut.app import event_loop
from multilabel_graphcut.common import MultiLabelState
from multilabel_graphcut.labels import read_labels


def main():
    parser = ArgumentParser()
    parser.add_argument("dir", type=Path)
    parser.add_argument("--filename-pattern", "-g", type=str, default="*.jpg")
    parser.add_argument("--out", "-o", type=Path, default=Path("output"))
    parser.add_argument("--label-config", "-c", type=Path, default=Path(__file__).parent / 'label_definitions' / 'human.json')
    args = parser.parse_args()

    labels = read_labels(args.label_config)

    # event loop
    for p in sorted(args.dir.rglob(args.filename_pattern)):
        save_dir = args.out / p.relative_to(args.dir).parent
        save_dir.mkdir(exist_ok=True, parents=True)
        save_label = save_dir / f"{p.stem}_label.png"
        save_image = save_dir / f"{p.stem}_image.png"
        save_mask = save_dir / f"{p.stem}_mask.png"

        if save_label.exists() and save_image.exists() and save_mask.exists():
            continue

        image = cv2.imread(p.as_posix())
        assert image is not None

        x, y, w, h = cv2.selectROI("annotation tool", image, showCrosshair=False)
        image = image[y: y + h, x: x + w]
        state = MultiLabelState.new(image)
        state = event_loop(state, labels)

        if state is None:
            p.unlink()
            continue
        cv2.imwrite(save_label.as_posix(), state.labelmap)
        cv2.imwrite(save_image.as_posix(), state.image)
        cv2.imwrite(save_mask.as_posix(), state.user_mask)
    print("Done!")

if __name__ == '__main__':
    main()
