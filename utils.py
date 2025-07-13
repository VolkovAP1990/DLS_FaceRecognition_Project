import cv2
import matplotlib.pyplot as plt

from skimage.transform import SimilarityTransform, warp
import torch.nn.functional as F
import torch
import numpy as np
import cv2
from PIL import Image

import torchvision.transforms as transforms
import pandas as pd

def get_face_bbox(image_rgb, detector_model, scale_w=0.2, scale_h=0.2, show=False, img_name="image.jpg"):
    """
    Получить масштабированный bbox лица из изображения с помощью YOLOv8-face.

    :param image_rgb: np.ndarray (H, W, 3) — изображение в RGB
    :param scale_w: float — масштаб по ширине (от bbox)
    :param scale_h: float — масштаб по высоте (от bbox)
    :param show: bool — показывать визуализацию bbox
    :param img_name: имя изображения (для заголовка)
    :return: (x1, y1, x2, y2) — bbox в координатах изображения
    """
    h_img, w_img = image_rgb.shape[:2]

    results = detector_model(image_rgb, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()

    if len(boxes) == 0:
        raise ValueError("Лицо не обнаружено.")

    best_idx = confidences.argmax()
    x1, y1, x2, y2 = map(int, boxes[best_idx])

    # Масштабируем bbox
    bbox_w = x2 - x1
    bbox_h = y2 - y1
    dw = int(bbox_w * scale_w)
    dh = int(bbox_h * scale_h)

    x1 = max(0, x1 - dw)
    x2 = min(w_img, x2 + dw)
    y1 = max(0, y1 - dh)
    y2 = min(h_img, y2 + dh)

    if show:
        print(f"[Scaled bbox] Width: {x2 - x1}, Height: {y2 - y1}")

        plt.figure(figsize=(4, 4))
        plt.imshow(image_rgb)
        plt.gca().add_patch(plt.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            edgecolor='blue', linewidth=2, fill=False, label="Face bbox"
        ))
        plt.title(f"YOLOv8-face bbox expanded — {img_name}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    return x1, y1, x2, y2


def predict_and_align(
    image_rgb,
    bbox,                 # ← Подаём bbox напрямую
    model,
    template,
    show=True,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    image_size=(160, 192)
):
    """
    Предсказание landmarks и выравнивание шаблону, используя bbox и RGB-изображение.

    :param image_rgb: np.ndarray (H, W, 3) — исходное RGB изображение
    :param bbox: [x1, y1, x2, y2] — координаты лица
    :param model: torch-модель
    :param template: шаблон ключевых точек (например, mean_template)
    :return: (aligned_image, aligned_landmarks)
    """
    x1, y1, x2, y2 = map(int, bbox)
    w, h = x2 - x1, y2 - y1

    # === Crop + resize
    image_crop = image_rgb[y1:y2, x1:x2]
    resized = cv2.resize(image_crop, image_size)  # (160x192)
    input_tensor = torch.tensor(resized.astype(np.float32) / 255.).permute(2, 0, 1).unsqueeze(0).to(device)

    # === Предсказание heatmaps
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)           # [1, N, 5, H, W]
        heatmaps_pred = output[:, -1]          # [1, 5, H, W]

    upsampled = F.interpolate(heatmaps_pred, size=(image_size[1], image_size[0]), mode='bilinear', align_corners=False)
    upsampled = upsampled.squeeze(0).cpu().numpy()  # [5, H, W]

    # === Извлекаем landmarks
    pred_landmarks = []
    for i in range(5):
        y_hm, x_hm = np.unravel_index(np.argmax(upsampled[i]), upsampled[i].shape)
        pred_landmarks.append([x_hm, y_hm])
    pred_landmarks = np.array(pred_landmarks, dtype=np.float32)

    # === Выравнивание изображения
    tform = SimilarityTransform()
    if not tform.estimate(pred_landmarks, template):
        raise ValueError("Не удалось найти similarity transform.")
    aligned = warp(resized, tform.inverse, output_shape=(image_size[1], image_size[0]), preserve_range=True).astype(np.uint8)

    aligned_landmarks = tform(pred_landmarks)

    # === Визуализация
    if show:
        vis1 = resized.copy()
        vis2 = aligned.copy()
        for (x_lm, y_lm) in pred_landmarks.astype(int):
            cv2.circle(vis1, (x_lm, y_lm), 3, (255, 0, 0), -1)
        for (x_lm, y_lm) in aligned_landmarks.astype(int):
            cv2.circle(vis2, (x_lm, y_lm), 3, (0, 255, 0), -1)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(vis1)
        plt.title("Before Alignment")
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(vis2)
        plt.title("Aligned Face")
        plt.axis('off')
        plt.show()

    return aligned


def predict_identity(image_np, img_name, model, df_val, device='cuda'):
    """
    Предсказывает класс лица по выровненному изображению и возвращает
    (предсказанный класс, уверенность, GT метка).

    :param image_np: np.ndarray — выровненное лицо (RGB)
    :param img_name: str — имя файла
    :param model: torch.nn.Module — модель ArcFace
    :param df_val: pd.DataFrame — таблица с GT метками (val.csv)
    :param device: str — устройство ('cuda' или 'cpu')
    :return: (pred_class, confidence, gt_label)
    """
    # === Трансформация
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(350),
        transforms.Pad((0, 0, 0, 50)),
        transforms.CenterCrop(224),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])

    image_pil = Image.fromarray(image_np)
    input_tensor = transform(image_pil).unsqueeze(0).to(device)

    # === Предсказание
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()

    # === GT метка
    row = df_val[df_val["img"] == img_name]
    if row.empty:
        raise ValueError(f"'{img_name}' не найден в таблице val.csv")
    gt_label = int(row["_id"].values[0])

    return pred_class, confidence, gt_label