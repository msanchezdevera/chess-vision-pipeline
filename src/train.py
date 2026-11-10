"""Fine-tuning de YOLOv8 Nano para detectar alfiles, caballos y torres."""
import argparse

import pandas as pd
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/data.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cpu", help="cpu, 0 (GPU), mps")
    args = parser.parse_args()

    model = YOLO("yolov8n.pt")
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
    )

    df = pd.read_csv(f"{results.save_dir}/results.csv")
    df.columns = df.columns.str.strip()
    last = df.iloc[-1]
    print("--- Métricas finales ---")
    print(f"Precision: {last['metrics/precision(B)']:.4f}")
    print(f"Recall:    {last['metrics/recall(B)']:.4f}")
    print(f"mAP50:     {last['metrics/mAP50(B)']:.4f}")
    print(f"mAP50-95:  {last['metrics/mAP50-95(B)']:.4f}")
    print(f"Pesos en:  {results.save_dir}/weights/best.pt")


if __name__ == "__main__":
    main()
