import os
import cv2
import random 
import numpy as np
from typing import Any, List, Tuple

import PIL
from PIL import Image, ImageFont, ImageDraw, ImageFilter

import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import DualTransform



def generate_random_image(
    fonts: List[str], 
    texts: List[str], 
    colors: List[Tuple[int, int, int]], 
    backgrounds: List[str], 
    opacities: List[float], 
    sizes: List[int],
    max_text_images: int = 5,
    final_size: Tuple[int, int] = (512, 512)
) -> Tuple[np.ndarray, np.ndarray]:

    """
    Генерирует случайное изображение и соответствующие тепловые карты.

    Параметры:
        fonts (List[str]): Список путей к файлам шрифтов.
        texts (List[str]): Список текстов для генерации.
        colors (List[Tuple[int, int, int]]): Список цветов текста в формате (R, G, B).
        backgrounds (List[str]): Список путей к изображениям фона.
        opacities (List[float]): Список значений прозрачности текста (0 - полностью прозрачный, 255 - полностью непрозрачный).
        sizes (List[int]): Список размеров шрифта.
        max_text_images int: Максимальное количество тектовых объектов на изображении.
        final_size (List[int, int]): Размер генерируемого изображения.

    Возвращает:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 
            - `image` (np.ndarray) — изображение с текстом.
            - `heatmap` (np.ndarray) — тепловая карта символов и связей (H, W, 2).
    """

    background_path = random.choice(backgrounds)
    background = Image.open(background_path).convert('RGB')
    background_np = np.array(background)

    background_np = augmentations_background(background_np)
    background = Image.fromarray(background_np)

    background = background.resize(final_size, Image.Resampling.BICUBIC)
    final_image = np.array(background)  # np.uint8, shape (H, W, 3)
    final_heatmap = np.zeros((final_size[1], final_size[0], 2), dtype=np.uint8)
    # Обратите внимание, что final_size = (W, H), а shape = (H, W), поэтому везде будьте аккуратны.

    text_pieces_count = random.randint(1, max_text_images)

    text_images, text_heatmaps = [], []
    for i in range(text_pieces_count):
        random_text = random.choice(texts)[:random.randint(0, 30)]
        random_font = random.choice(fonts)
        random_color = random.choice(colors)
        random_opacity = random.choice(opacities)
        random_size = random.choice(sizes)
        
        piece_img, char_mask, aff_mask = generate_text_image(
            text=random_text,
            font_path=random_font,
            color=random_color,
            font_size=random_size,
            opacity=random_opacity
        )

        piece_img_np = np.array(piece_img)
        hmap = np.stack([char_mask, aff_mask], axis=-1)

        if text_pieces_count == 1:
            angle = random.randint(-70, 70)
        elif text_pieces_count == 2:
            angle = random.randint(-30, 30)
        elif text_pieces_count == 3:
            angle = random.randint(-15, 15)
        else :
            angle = random.randint(-5, 5)
        
        piece_img_np, hmap = rotate_piece(piece_img_np, hmap, angle)

        if text_pieces_count == 1:
            h, w = piece_img_np.shape[:2]
            scale_factor = final_size[0] / float(w)
        else:
            scale_factor = random.uniform(0.3, 0.6)

        piece_img_np, hmap = scale_piece(piece_img_np, hmap, scale_factor)

        text_images.append(piece_img_np)
        text_heatmaps.append(hmap)


    placed_boxes = []
    final_h, final_w = final_size[1], final_size[0]

    for piece_img, piece_hmap in zip(text_images, text_heatmaps):
        ph, pw = piece_img.shape[:2]

        if pw > final_w or ph > final_h:
           
            forced_scale_factor = min(final_w / float(pw), final_h / float(ph)) * 0.95
            if forced_scale_factor <= 0.1:
                continue

            piece_img, piece_hmap = scale_piece(piece_img, piece_hmap, forced_scale_factor)
            ph, pw = piece_img.shape[:2]

        for _ in range(30):
            x = random.randint(0, final_w - pw)
            y = random.randint(0, final_h - ph)
            candidate_box = (x, y, x + pw, y + ph)
            if not intersects_any(candidate_box, placed_boxes):
                overlay_with_alpha(final_image, piece_img, x, y)
                merge_heatmaps(final_heatmap, piece_hmap, x, y)
                placed_boxes.append(candidate_box)
                break
        else:
            pass

    character_mask = np.array(Image.fromarray(final_heatmap[..., 0]).convert('L').filter(ImageFilter.GaussianBlur(radius=3)))
    affinity_mask = np.array(Image.fromarray(final_heatmap[..., 1]).convert('L').filter(ImageFilter.GaussianBlur(radius=3)))
    
    final_heatmap = np.stack([character_mask, affinity_mask], axis=2)
    final_image, final_heatmap = augmentations_image(final_image, final_heatmap)
    
    return final_image, final_heatmap



def save_image_and_mask(
    idx: int,
    image: np.ndarray, 
    heatmap: np.ndarray, 
    images_folder: str, 
    char_masks_folder: str, 
    aff_mask_folder: str
) -> None:
    
    """
    Сохраняет изображение и маски символов и связей в указанные папки.

    Параметры:
        idx (int): Индекс изображения, используемый в имени файла.
        image (np.ndarray): Массив изображения (обычно 3D: (H, W, C)).
        heatmap (np.ndarray): Массив с тепловыми картами (обычно 3D: (H, W, 2)), 
            где первый канал - character_mask, второй - affinity_mask.
        images_folder (str): Путь к папке для сохранения изображений.
        char_masks_folder (str): Путь к папке для сохранения character_mask.
        aff_mask_folder (str): Путь к папке для сохранения affinity_mask.

    Возвращает:
        None
    """
    try:
        image = Image.fromarray(image)

        character_mask = Image.fromarray(heatmap[..., 0]).convert('L')
        affinity_mask = Image.fromarray(heatmap[..., 1]).convert('L')
        
        image.save(os.path.join(images_folder, f'image_{idx}.png'))
        character_mask.save(os.path.join(char_masks_folder, f'character_heatmap_{idx}.png'))
        affinity_mask.save(os.path.join(aff_mask_folder, f'affinity_heatmap_{idx}.png'))

    except Exception as e:
        print(f"Ошибка при сохранении {idx}: {e}")



def generate_text_image(
    text: str, 
    font_path: str, 
    font_size: int = 64, 
    draw_points: bool = False, 
    scale: float = 1.0, 
    interval: int = 0, 
    color: Tuple[int, int, int] = (0, 0, 0), 
    opacity: int = 255
) -> Tuple[PIL.Image.Image, np.ndarray, np.ndarray]:
    
    """
    Создаёт изображение строки с текстом и соответствующими масками.

    Параметры:
        text (str): Строка генерируемого текста.
        font_path (str): Пусть до используемого в генерации шрифта.
        font_size (int): Размер генерируемого текста, 
        draw_points (bool): Флаг отрисовки на строке текста геометрических точек.
        scale (float): Масштаб увеличения bbox символа для вставки изображения Гауссианы.
        interval (int): Значение увеличения интервала между символами.
        color (list(int, int, int)): Цвет генерируемого текста.
        opacity (int): Значение непрозрачности слоя текста.

    Возвращает:
        Tuple[PIL.Image.Image, np.ndarray, np.ndarray]: 
            - `image` (PIL.Image.Image) — изображение RGBA с текстом.
            - `character_mask` (np.ndarray) — тепловая карта символов (H, W).
            - `affinity_mask` (np.ndarray) — тепловая карта межсимволов (H, W).
    """

    font = ImageFont.truetype(font_path, size=font_size)
    left, top, right, bottom = font.getbbox(text)

    if interval != 0:
        counts = len(text) - 2
        right += counts * interval

    padding = font_size * 5
    right += padding
    bottom += 10
    
    image_size = (right, bottom)
    
    image = Image.new('RGBA', image_size, (0, 0, 0, 0))
    character_mask = np.zeros((bottom, right), dtype=np.uint8)
    affinity_mask = np.zeros((bottom, right), dtype=np.uint8)

    draw_image = ImageDraw.Draw(image)
    
    prev_x, prev_y = 20, 0
    
    gaussian_size = 200
    gaussian = generate_gaussian(size=gaussian_size, sigma=0.4, crop=10)
    text_color = tuple(list(color) + [opacity])

    character_boxes, affinity_points = [], []
    for char in text:
        x0, y0, x1, y1 = font.getbbox(char)
        width, height = x1 - x0, y1 - y0
        
        if char == ' ':
            prev_x += width
            affinity_points.append(-1)
            continue  
    
        bbox = (x0 + prev_x, y0 + prev_y, x1 + prev_x, y1 + prev_y)

        # Увеличиваем bbox на scale
        dx, dy = width * (scale - 1) / 2, height * (scale - 1) / 2
        new_bbox = (bbox[0] - dx, bbox[1] - dy, bbox[2] + dx, bbox[3] + dy)

        center_char_x = (bbox[2] - bbox[0]) // 2 + bbox[0]
        center_char_y = (bbox[3] - bbox[1]) // 2 + bbox[1]

        top_triangle = [(bbox[0], bbox[1]), (center_char_x, center_char_y), (bbox[2], bbox[1])]
        bottom_triangle = [(bbox[0], bbox[3]), (center_char_x, center_char_y), (bbox[2], bbox[3])]

        character_boxes.append(bbox)
        affinity_points.append((top_triangle, bottom_triangle))      

        # Перспективное преобразование гауссианы под размер символа
        square_points = np.array([[0, 0], [gaussian_size, 0], [gaussian_size, gaussian_size], [0, gaussian_size]], dtype=np.float32)
        target_points = np.array([
            [new_bbox[0], new_bbox[1]], [new_bbox[2], new_bbox[1]],
            [new_bbox[2], new_bbox[3]], [new_bbox[0], new_bbox[3]]], dtype=np.float32)
        
        M = cv2.getPerspectiveTransform(square_points, target_points)
        warped_gaussian = cv2.warpPerspective(gaussian, M, (image_size[0], image_size[1]))

        mask = warped_gaussian > character_mask
        character_mask[mask] = warped_gaussian[mask]
        
        if draw_points:
            draw_image.rectangle(bbox, outline='green', width=1)
        draw_image.text((prev_x, prev_y), char, font=font, fill=text_color)

        if interval != 0:
            prev_x += width + interval
        else:
            prev_x += width
    
    # Формируем affinity-маску
    for i in range(len(affinity_points) - 1):
        cur = affinity_points[i]
        nxt = affinity_points[i + 1]
        if cur == -1 or nxt == -1:
            continue
        # cur, nxt содержат треугольники (top, bottom)
        # Для простоты генерируем гауссиану между нижней частью одного символа и верхней частью следующего
        # Но в вашем коде своя логика, оставим всё, как есть, по аналогии
        top_triangle_cur, bottom_triangle_cur = cur
        top_triangle_nxt, bottom_triangle_nxt = nxt

        # Вычислим центры
        top_center_cur = get_triangle_center(top_triangle_cur)
        bottom_center_cur = get_triangle_center(bottom_triangle_cur)
        top_center_nxt = get_triangle_center(top_triangle_nxt)
        bottom_center_nxt = get_triangle_center(bottom_triangle_nxt)

        # 4 точки для affinity
        polygon = [bottom_center_cur, bottom_center_nxt, top_center_nxt, top_center_cur]

        # Увеличим полигон
        polygon_scaled = scale_polygon(polygon, scale=1.2)

        # Перспективная трансформация
        gaussian_size = 200
        gaussian = generate_gaussian(size=gaussian_size, sigma=0.4, crop=10)

        square_pts = np.array([[0, 0], [gaussian_size, 0], [gaussian_size, gaussian_size], [0, gaussian_size]], dtype=np.float32)
        target_pts = np.array(polygon_scaled, dtype=np.float32)

        M = cv2.getPerspectiveTransform(square_pts, target_pts)
        warped_gaussian = cv2.warpPerspective(gaussian, M, (image_size[0], image_size[1]))

        mask = warped_gaussian > affinity_mask
        affinity_mask[mask] = warped_gaussian[mask]

    piece_width = prev_x + 20
    piece_height = bottom
    image = image.crop((0, 0, piece_width, piece_height))
    character_mask = character_mask[:piece_height, :piece_width]
    affinity_mask = affinity_mask[:piece_height, :piece_width]

    return image, character_mask, affinity_mask



def augmentations_image(image, mask):
    geometry_augmentations = A.Compose([
        A.Perspective(scale=(0.0, 0.001), p=0.5, keep_size=False),
        A.Rotate(limit=5, border_mode=cv2.BORDER_REFLECT, p=0.5),
    ], additional_targets={'mask': 'mask'})
    
    appearance_augmentations = A.Compose([
        A.GaussNoise(std_range=(0.01, 0.05), p=0.5),
        A.MedianBlur(blur_limit=3, p=0.2), 
        A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        A.HueSaturationValue(p=1)        
    ])

    augmented = geometry_augmentations(image=image, mask=mask)
    image, mask = augmented['image'], augmented['mask']
    
    augmented = appearance_augmentations(image=image)
    image = augmented['image']

    return image, mask


def augmentations_background(background):
    augmentations = A.Compose([
        A.HorizontalFlip(p=0.5),  # Отражение по горизонтали
        A.Perspective(p=0.5, keep_size=True),  # Перспективные искажения
        A.Rotate(limit=30, border_mode=cv2.BORDER_REFLECT, p=0.5),
        A.GaussNoise(std_range=(0.01, 0.05), p=0.5),  # Шум
        A.MedianBlur(blur_limit=3, p=0.2),  # Размытие медианным фильтром
        A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.5),  # Случайное изменение гаммы
        A.HueSaturationValue(p=0.5)
    ])
    
    return augmentations(image=background)['image']


def rotate_piece(piece_img: np.ndarray, heatmap: np.ndarray, angle: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Поворачивает кусок (RGBA или RGB) и соответствующий heatmap
    на заданный угол (в градусах).
    """
    # piece_img shape: (H, W, 4) (если RGBA) или (H, W, 3) (если RGB)
    # heatmap shape: (H, W, 2)

    pil_img = Image.fromarray(piece_img)
    rotated_img = pil_img.rotate(angle, expand=True)
    rotated_img_np = np.array(rotated_img)

    pil_ch0 = Image.fromarray(heatmap[..., 0])
    pil_ch1 = Image.fromarray(heatmap[..., 1])
    rot_ch0 = pil_ch0.rotate(angle, expand=True)
    rot_ch1 = pil_ch1.rotate(angle, expand=True)
    ch0_np = np.array(rot_ch0)
    ch1_np = np.array(rot_ch1)

    heatmap_new = np.stack([ch0_np, ch1_np], axis=-1)
    return rotated_img_np, heatmap_new


def scale_piece(piece_img: np.ndarray, heatmap: np.ndarray, scale_factor: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Масштабирует кусок изображения piece_img и heatmap
    """
    h, w = piece_img.shape[:2]
    new_w = max(1, int(w * scale_factor))
    new_h = max(1, int(h * scale_factor))

    # Масштабируем картинку
    pil_img = Image.fromarray(piece_img)
    pil_img = pil_img.resize((new_w, new_h), Image.Resampling.BICUBIC)
    piece_img_scaled = np.array(pil_img)

    # Масштабируем каналы heatmap
    ch0 = Image.fromarray(heatmap[..., 0])
    ch1 = Image.fromarray(heatmap[..., 1])
    ch0 = ch0.resize((new_w, new_h), Image.Resampling.BICUBIC)
    ch1 = ch1.resize((new_w, new_h), Image.Resampling.BICUBIC)

    hmap_scaled = np.stack([np.array(ch0), np.array(ch1)], axis=-1)

    return piece_img_scaled, hmap_scaled


def intersects_any(box: Tuple[int, int, int, int], boxes: List[Tuple[int, int, int, int]]) -> bool:
    """
    Проверяет, пересекается ли box с любым из списка boxes.
    box в формате (x1, y1, x2, y2).
    """
    x1, y1, x2, y2 = box
    for xb1, yb1, xb2, yb2 in boxes:
        # Горизонтальное пересечение?
        overlap_x = not (x2 < xb1 or x1 > xb2)
        # Вертикальное пересечение?
        overlap_y = not (y2 < yb1 or y1 > yb2)
        if overlap_x and overlap_y:
            return True
    return False


def overlay_with_alpha(base_img: np.ndarray, overlay_img: np.ndarray, x: int, y: int) -> None:
    """
    Накладывает overlay_img (RGBA) на base_img (RGB) 
    с учётом альфа-канала, начиная с координат (x,y) в base_img.
    Изменяет base_img на месте.
    """
    if overlay_img.shape[-1] == 3:
        # Нет альфа-канала, просто накладываем как есть
        oh, ow = overlay_img.shape[:2]
        base_img[y:y+oh, x:x+ow] = overlay_img
        return

    # RGBA вариант
    oh, ow = overlay_img.shape[:2]
    alpha_s = overlay_img[:, :, 3] / 255.0
    for c in range(3):  # для R, G, B
        base_img[y:y+oh, x:x+ow, c] = (
            alpha_s * overlay_img[:, :, c] + 
            (1 - alpha_s) * base_img[y:y+oh, x:x+ow, c]
        )


def merge_heatmaps(base_heatmap: np.ndarray, overlay_heatmap: np.ndarray, x: int, y: int) -> None:
    """
    Простейшее объединение heatmap'ов.
    Можно делать np.maximum, np.add, усреднение и т.д.
    Меняем base_heatmap на месте.
    """
    oh, ow = overlay_heatmap.shape[:2]
    region = base_heatmap[y:y+oh, x:x+ow, :]
    # например, возьмём максимум
    base_heatmap[y:y+oh, x:x+ow, :] = np.maximum(region, overlay_heatmap)


def generate_gaussian(size=30, sigma=1., crop=False):
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    x, y = np.meshgrid(x, y)
    gaussian = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    gaussian = (gaussian * 255).astype(np.uint8)
    if crop:
        gaussian[gaussian < crop] = 0
    return gaussian


def get_triangle_center(triangle):
    x_center = sum(p[0] for p in triangle) // 3
    y_center = sum(p[1] for p in triangle) // 3
    return x_center, y_center


def scale_polygon(polygon, scale=1.1):
    polygon = np.array(polygon, dtype=np.float32)
    center = np.mean(polygon, axis=0)
    scaled_polygon = center + (polygon - center) * scale 
    return [tuple(point) for point in scaled_polygon]


def overlay_with_alpha(base_img, overlay_img, x, y):
    oh, ow = overlay_img.shape[:2]
    bh, bw = base_img.shape[:2]

    # Если фрагмент «вылез» за правый или нижний край:
    max_x = x + ow
    max_y = y + oh

    # Обрезаем overlay_img, если выходит за границы
    # 1) Вычислим, какие координаты реально влезают
    if max_x > bw:  # выходит за правый край
        ow = bw - x  # ширина, которая влезет
    if max_y > bh:  # выходит за нижний край
        oh = bh - y  # высота, которая влезет

    if ow <= 0 or oh <= 0:
        # вообще не помещается
        return

    # Обрезаем overlay_img (само изображение и альфа-канал) 
    # только если нужно:
    overlay_cropped = overlay_img[:oh, :ow, :]

    alpha_s = overlay_cropped[:, :, 3] / 255.0  # (oh, ow)
    # Далее обычная схема:
    for c in range(3):
        base_img[y:y+oh, x:x+ow, c] = (
            alpha_s * overlay_cropped[:, :, c] +
            (1 - alpha_s) * base_img[y:y+oh, x:x+ow, c]
        )
