"""Extrae un fotograma cada N segundos de un video para armar el dataset."""
import argparse
from pathlib import Path

import cv2


def extract_frames(video: Path, output: Path, every_seconds: float) -> int:
    output.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir el video: {video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    interval = max(1, int(fps * every_seconds))
    frame_idx, saved = 0, 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % interval == 0:
            cv2.imwrite(str(output / f"chess_frame_{saved:03d}.jpg"), frame)
            saved += 1
        frame_idx += 1

    cap.release()
    return saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/raw_frames"))
    parser.add_argument("--every", type=float, default=10.0, help="segundos entre fotogramas")
    args = parser.parse_args()
    n = extract_frames(args.video, args.output, args.every)
    print(f"Se guardaron {n} imágenes en {args.output}")
