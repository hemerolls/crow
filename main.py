"""
Crowd Detection with YOLOv8

Этот скрипт выполняет детекцию людей на видеофайле crowd.mp4,
отрисовывает bounding boxes вокруг них и сохраняет результат в output_tracked.mp4.
"""

import cv2
import numpy as np
from ultralytics import YOLO


def load_model(model_name: str = "yolov8l.pt") -> YOLO:
    """
    Загружает предобученную модель YOLO.

    Args:
        model_name (str): Имя модели YOLO для загрузки.

    Returns:
        YOLO: Загруженная модель.
    """
    return YOLO(model_name)


def process_video(video_path: str, output_path: str):
    """
    Обрабатывает видео: обнаруживает людей и сохраняет результат.

    Args:
        video_path (str): Путь к входному видеофайлу.
        output_path (str): Путь для сохранения выходного видео.

    Returns:
        None
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Не удалось открыть видеофайл")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    print(f"Видео считывается. FPS: {fps}, Размер кадра: {width}x{height}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        raise ValueError("Не удалось создать VideoWriter")

    model = load_model()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.6, iou=0.45, verbose=False)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = box.cls.item()
                conf = box.conf.item()

                if cls == 0 and conf > 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Рисуем прямоугольник
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (127, 255, 0), 2)

                    # Подпись с уверенностью
                    label = f"person {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()
    print("✅ Обработка завершена.")


if __name__ == "__main__":
    input_video = "crowd.mp4"
    output_video = "output/output_tracked.mp4"

    # Создаём папку вывода, если её нет
    os.makedirs(os.path.dirname(output_video), exist_ok=True)

    process_video(input_video, output_video)