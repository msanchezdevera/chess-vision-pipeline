"""Pipeline unificado de visión por computadora sobre una partida de ajedrez.

Combina tres modelos YOLOv8 sobre el mismo video:
  A. Detección custom de piezas (alfil, caballo, torre) + NMS + conteo.
  B. Segmentación del entorno (COCO), "perforada" donde están las piezas.
  C. Estimación de pose de los jugadores, dibujada por encima de todo.
"""
import argparse

import cv2
import numpy as np
from ultralytics import YOLO

CLASSES = ("Alfil", "Caballo", "Torre")


def detect_pieces(model, frame, conf_th, nms_th):
    boxes, confs, cls_ids = [], [], []
    for r in model(frame, verbose=False):
        if r.boxes is None:
            continue
        for box, cls, conf in zip(r.boxes.xyxy.cpu().numpy(),
                                  r.boxes.cls.cpu().numpy(),
                                  r.boxes.conf.cpu().numpy()):
            if conf > conf_th:
                x1, y1, x2, y2 = map(int, box[:4])
                boxes.append([x1, y1, x2 - x1, y2 - y1])
                confs.append(float(conf))
                cls_ids.append(int(cls))

    pieces, counts = [], {c: 0 for c in CLASSES}
    keep = cv2.dnn.NMSBoxes(boxes, confs, score_threshold=conf_th, nms_threshold=nms_th)
    for i in np.array(keep).flatten():
        x1, y1, w, h = boxes[i]
        name = model.names[cls_ids[i]]
        if name in counts:
            counts[name] += 1
            pieces.append((x1, y1, x1 + w, y1 + h, name, confs[i]))
    return pieces, counts


def draw_environment(model, frame, pieces):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for r in model(frame, verbose=False):
        if r.masks is None or r.boxes is None:
            continue
        for poly, cls in zip(r.masks.xy, r.boxes.cls.cpu().numpy()):
            if model.names[int(cls)] == "person":
                continue
            cv2.fillPoly(mask, [np.array(poly, dtype=np.int32)], 255)

    for x1, y1, x2, y2, *_ in pieces:
        cv2.rectangle(mask, (x1, y1), (x2, y2), 0, -1)

    overlay = frame.copy()
    overlay[mask == 255] = (0, 255, 0)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    for x1, y1, x2, y2, name, conf in pieces:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
        cv2.putText(frame, f"{name} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
    return frame


def draw_pose(model, frame):
    for r in model(frame, verbose=False):
        if r.keypoints is not None:
            frame = r.plot(boxes=False, masks=False)
    return frame


def draw_panel(frame, counts):
    cv2.rectangle(frame, (10, 10), (240, 120), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (240, 120), (255, 255, 255), 2)
    cv2.putText(frame, "Piezas en juego:", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    for i, name in enumerate(CLASSES):
        cv2.putText(frame, f"- {name}: {counts[name]}", (20, 60 + 20 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return frame


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video")
    p.add_argument("--output", default="resultado.mp4")
    p.add_argument("--weights", default="models/best.pt")
    p.add_argument("--skip", type=int, default=3, help="procesa 1 de cada N cuadros")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--nms", type=float, default=0.4)
    p.add_argument("--show", action="store_true", help="muestra el video en vivo")
    args = p.parse_args()

    model_custom = YOLO(args.weights)
    model_seg = YOLO("yolov8n-seg.pt")
    model_pose = YOLO("yolov8n-pose.pt")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir el video: {args.video}")
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = max(1, int(cap.get(cv2.CAP_PROP_FPS) / args.skip))
    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if idx % args.skip:
                continue

            pieces, counts = detect_pieces(model_custom, frame, args.conf, args.nms)
            frame = draw_environment(model_seg, frame, pieces)
            frame = draw_pose(model_pose, frame)
            frame = draw_panel(frame, counts)
            out.write(frame)

            if args.show:
                cv2.imshow("Chess Vision Pipeline", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    print(f"Video procesado guardado en {args.output}")


if __name__ == "__main__":
    main()
