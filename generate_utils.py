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
    sizes: List[int]
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

    Возвращает:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 
            - `image` (np.ndarray) — изображение с текстом.
            - `heatmap` (np.ndarray) — тепловая карта символов и связей (H, W, 2).
    """
    
    images = []
    heatmaps = []

    for row in range(np.random.randint(3, 5)):
        random_text = ' '.join(random.choice(texts).split(' ')[:4])
        random_font = random.choice(fonts)
        random_color = random.choice(colors)
        random_background = random.choice(backgrounds)
        random_opacity = random.choice(opacities)
        random_size = random.choice(sizes)
        random_angle = random.randint(-15, 15)
        
        image, character_mask, affinity_mask = generate_text_image(
            text=random_text, 
            font_path=random_font, 
            color=random_color, 
            font_size=random_size, 
            opacity=random_opacity
        )

        image = image.rotate(random_angle, expand=True)
        character_mask = np.array(Image.fromarray(character_mask).rotate(random_angle, expand=True))
        affinity_mask = np.array(Image.fromarray(affinity_mask).rotate(random_angle, expand=True))

        try:
            width, height = image.size
            
            heatmap = np.zeros((height, width, 2))
            heatmap[..., 0] = character_mask
            heatmap[..., 1] = affinity_mask
    
            image = np.array(image)
            
            images.append(image)
            heatmaps.append(heatmap)
            
        except:
            continue

    image, heatmap = merge_multiline(images, heatmaps)
    image, heatmap = shift(image, heatmap=heatmap)

    background = np.array(Image.open(random_background))
    background, _ = image_augmentation(background, geomerty_transform=True, appearance_transform=True)
    background = Image.fromarray(background)

    image = add_paper(image, background, margins=(0, 0))

    image, heatmap = image_augmentation(image, heatmap, geomerty_transform=True, appearance_transform=True, final=True)

    character_mask = np.array(Image.fromarray(heatmap[..., 0]).convert('L').filter(ImageFilter.GaussianBlur(radius=3)))
    affinity_mask = np.array(Image.fromarray(heatmap[..., 1]).convert('L').filter(ImageFilter.GaussianBlur(radius=3)))

    heatmap = np.stack([character_mask, affinity_mask], axis=2)

    return image, heatmap



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
    character_mask = Image.new('RGBA', image_size, (0, 0, 0, 0))
    affinity_mask = Image.new('RGBA', image_size, (0, 0, 0, 0))

    character_mask = np.zeros((bottom, right))
    affinity_mask = np.zeros((bottom, right))

    draw_image = ImageDraw.Draw(image)
    
    prev_x, prev_y = 20, 0
    
    gaussian_size = 200
    gaussian = generate_gaussian(size=gaussian_size, sigma=0.4, crop=10)
    color = tuple(color + [opacity])

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

        top_center = get_triangle_center(top_triangle)
        bottom_center = get_triangle_center(bottom_triangle)

        character_boxes.append(bbox)
        affinity_points.append((top_center, bottom_center))      

        square_points = np.array([[0, 0], [gaussian_size, 0], [gaussian_size, gaussian_size], [0, gaussian_size]], dtype=np.float32)
        target_points = np.array([
            [new_bbox[0], new_bbox[1]], [new_bbox[2], new_bbox[1]],
            [new_bbox[2], new_bbox[3]], [new_bbox[0], new_bbox[3]]], dtype=np.float32)
        
        M = cv2.getPerspectiveTransform(square_points, target_points)
        warped_gaussian = cv2.warpPerspective(gaussian, M, (image_size[0], image_size[1]))

        mask = warped_gaussian > character_mask
        character_mask[mask] = warped_gaussian[mask]
        

        if draw_points:
            draw_image.rectangle(bbox, outline='green', width=3)
            draw_image.ellipse([center_char_x - 2, center_char_y - 2, center_char_x + 2, center_char_y + 2], fill='green', width=3)
            draw_image.polygon(top_triangle, outline='blue', width=3)
            draw_image.polygon(bottom_triangle, outline='blue', width=3)
            draw_image.ellipse([top_center[0] - 2, top_center[1] - 2, top_center[0] + 2, top_center[1] + 2], fill="blue", width=3)
            draw_image.ellipse([bottom_center[0] - 2, bottom_center[1] - 2, bottom_center[0] + 2, bottom_center[1] + 2], fill="blue", width=3)

        draw_image.text((prev_x, prev_y), char, font=font, fill=color)

        if interval != 0:
            prev_x += width + interval
        else:
            prev_x += width
    
    affinity_polygons = []
    for i in range(len(affinity_points) - 1):
        current_char = affinity_points[i]
        next_char = affinity_points[i + 1]
        
        if next_char == -1:
            continue
        
        if current_char != -1:
            polygon = (current_char[0], current_char[1], next_char[1], next_char[0])
            affinity_polygons.append(polygon)

            new_polygon = scale_polygon(polygon, scale=scale+0.2)
            
            mean_x = np.mean([current_char[0][0], current_char[1][0], next_char[1][0], next_char[0][0]])
            mean_y = np.mean([current_char[0][1], current_char[1][1], next_char[1][1], next_char[0][1]])

            # Определяем 4 точки квадрата для перспективного трансформа
            square_points = np.array([[0, 0], [gaussian_size, 0], [gaussian_size, gaussian_size], [0, gaussian_size]], dtype=np.float32)
            target_points = np.array(new_polygon, dtype=np.float32)

            # Вычисляем матрицу трансформации
            M = cv2.getPerspectiveTransform(square_points, target_points)

            # Применяем перспективное преобразование
            warped_gaussian = cv2.warpPerspective(gaussian, M, (image_size[0], image_size[1]))

            mask = warped_gaussian > affinity_mask
            affinity_mask[mask] = warped_gaussian[mask]

        
    image = image.crop((0, 0, prev_x + 20, bottom))
    character_mask = character_mask[:, : prev_x + 20]
    affinity_mask = affinity_mask[:, : prev_x + 20]
    

    return image, character_mask, affinity_mask



def generate_gaussian(size=30, sigma=1., crop=False):
    """
    Создаёт квадратную гауссиану.
    """
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    x, y = np.meshgrid(x, y)
    gaussian = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    gaussian = (gaussian * 255).astype(np.uint8)
    if crop: gaussian[gaussian < crop] = 0
    return gaussian


def get_triangle_center(triangle):
    x_center = sum(p[0] for p in triangle) // 3
    y_center = sum(p[1] for p in triangle) // 3
    return x_center, y_center


def scale_polygon(polygon, scale=1.1):
    """
    Увеличивает полигон на заданный коэффициент относительно его центра.
    
    :param polygon: Список кортежей (x, y) вершин полигона
    :param scale: Коэффициент увеличения
    :return: Новый полигон с увеличенными координатами
    """
    polygon = np.array(polygon, dtype=np.float32)
    center = np.mean(polygon, axis=0)  # Находим центр полигона
    
    # Увеличиваем расстояние от центра до каждой точки
    scaled_polygon = center + (polygon - center) * scale 
    return [tuple(point) for point in scaled_polygon]



def cvt2HeatmapImg(img):
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    return img
    
def add_heatmap2img(img,heatmap):
    color_heatmap = cvt2HeatmapImg(heatmap)
    return cv2.addWeighted(color_heatmap, 0.5,img ,0.5,0)

def merge_multiline(imgs: list, heatmaps: list) -> tuple[np.ndarray, np.ndarray]:
    spacings = []
    width, height = 0, 0
    for img in imgs:
        img_h, img_w, channels = img.shape
        
        spacing = np.random.randint(5,10+1) # change TODO
        if img_h + spacing < 0:
            spacing = 0
        spacings.append(spacing)

        width = max(width, img_w)
        height += img_h
        height += spacing
    height -= spacing

    combined_img = np.zeros((height, width, channels), dtype="uint16")
    combined_heatmap = np.zeros((height, width, 2), dtype="float64")
    y_start = 0
    for i, (img, heatmap) in enumerate(zip(imgs, heatmaps)):
        img_h, img_w, _ = img.shape
        x_start = np.random.randint(0, width - img_w +1)
        combined_img[y_start:y_start + img_h, x_start:x_start + img_w, :] += img
        combined_heatmap[y_start:y_start + img_h, x_start:x_start + img_w, :] += heatmap
        y_start += img_h + spacings[i]
    combined_img = np.clip(combined_img, 0, 255).astype("uint8")

    return combined_img, combined_heatmap

def shift(img: np.ndarray, heatmap: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    margins = (15,15) # TODO
    pad_left = np.random.randint(margins[0]) if margins[0] > 0 else 0
    pad_top = np.random.randint(margins[1]) if margins[1] > 0 else 0
    pad_right = np.random.randint(margins[0]) if margins[0] > 0 else 0
    pad_bottom = np.random.randint(margins[1]) if margins[1] > 0 else 0

    h, w, _ = heatmap.shape
    new_w = pad_left + w + pad_right
    new_h = pad_top + h + pad_bottom

    shifted_new_img = np.zeros((new_h, new_w, img.shape[-1]), dtype="uint8")
    shifted_new_img[pad_top:h + pad_top, pad_left:w + pad_left, :] += img

    shifted_heatmap = np.zeros((new_h, new_w, 2))
    shifted_heatmap[pad_top:h + pad_top, pad_left:w + pad_left] += heatmap

    return shifted_new_img, shifted_heatmap

def pad_to_size(img: Image.Image, size: tuple):
    new_img = Image.new("RGBA", size, (255, 255, 255, 0))
    x_marg = (size[0] - img.size[0]) // 2
    y_marg = (size[1] - img.size[1]) // 2
    new_img.paste(img, (x_marg, y_marg))
    return new_img
    
def fetch_paper_fragment(img: Image.Image, back: Image.Image, margins: tuple[int, int] = (10, 10)):
    back = back.convert("RGBA")
    size = tuple([d + margin * 2 for d, margin in zip(img.size, margins)])
    if back.size[0] < size[0] or back.size[1] < size[1]:
        back = back.resize(size, Image.Resampling.BICUBIC)
    xrange = back.size[0] - size[0]
    yrange = back.size[1] - size[1]
    x = random.randint(0, xrange)
    y = random.randint(0, yrange)
    return back.crop((x, y, x + size[0], y + size[1]))
    
def add_paper(img: np.ndarray, paper_path: Image.Image, margins: tuple[int, int] = None) -> np.ndarray:
    img = Image.fromarray(img)
    margins = margins if margins is not None else margins
    paper = fetch_paper_fragment(img, back=paper_path, margins=margins)
    img = pad_to_size(img, paper.size)
    img = Image.alpha_composite(paper, img)
    img = np.array(img)
    return img


def image_augmentation(image, mask=None, geomerty_transform=False, appearance_transform=False, final=False):
    geometry_augmentations = A.Compose([
        # A.HorizontalFlip(p=0.5),  # Отражение по горизонтали
        A.Perspective(scale=(0.0, 0.001), p=0.5, keep_size=False),  # Перспективные искажения
        A.Rotate(limit=5, border_mode=cv2.BORDER_REFLECT, p=0.5),
    ], additional_targets={'mask': 'mask'})  # Указываем, что маска трансформируется так же
    
    appearance_augmentations = A.Compose([
        A.GaussNoise(std_range=(0.01, 0.05), p=0.5),  # Шум
        A.MedianBlur(blur_limit=3, p=0.2),  # Размытие медианным фильтром
        A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.5),  # Случайное изменение гаммы
        A.HueSaturationValue(p=0.5)        
    ])

    if final:
        image = np.array(Image.fromarray(image).convert('RGB'))

    if geomerty_transform:
        augmented = geometry_augmentations(image=image, mask=mask)
        image, mask = augmented['image'], augmented['mask']
        
    if appearance_transform:
        augmented = appearance_augmentations(image=image)
        image = augmented['image']

    return image, mask






        