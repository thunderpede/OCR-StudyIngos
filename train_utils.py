import gc
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
import concurrent.futures

from tqdm import tqdm
from PIL import Image, ImageDraw

import torch
import torch.nn as nn

from torch.utils.data import Dataset
from typing import Dict, Any, List, Tuple, Optional

from loss_functions import DiceFocalLoss

import craft_utils

from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.segformer.modeling_segformer import SegformerPreTrainedModel, SegformerModel, SegformerDecodeHead

from torchvision.transforms import ToPILImage


class SegformerForСraft(SegformerPreTrainedModel):
    def __init__(self, config, loss_fn=None):
        super().__init__(config)
        
        self.segformer = SegformerModel(config)
        self.decode_head = SegformerDecodeHead(config)
        self.loss_fn = loss_fn if loss_fn else DiceFocalLoss()
        self.init_weights()

    def forward(self, pixel_values, labels=None, output_attentions=None, 
        output_hidden_states=None, return_dict=None):
        
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.segformer(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=return_dict,
        )

        encoder_hidden_states = outputs.hidden_states if return_dict else outputs[1]

        logits = self.decode_head(encoder_hidden_states)

        loss = None
        if labels is not None:
            if self.config.num_labels == 1:
                raise ValueError("The number of labels should be greater than one")
            else:
                upsampled_logits = nn.functional.interpolate(
                    logits, size=labels.shape[-2:], 
                    mode="bilinear", align_corners=False
                )
                
                loss = self.loss_fn(upsampled_logits, labels)

        if not return_dict:
            if output_hidden_states:
                output = (logits,) + outputs[1:]
            else:
                output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=torch.clip(logits,0,1),
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions
        )


class CRAFTDataset(Dataset):
    """
    Датасет для обучения модели CRAFT (Character Region Awareness for Text detection).
    Загружает изображения и соответствующие тепловые карты (heatmaps) для character heatmap и affinity heatmap.
    
    Атрибуты:
        feature_extractor: Объект, выполняющий предобработку изображений.
        images (List[str]): Пути к изображениям.
        character_heatmaps (List[str]): Пути к character heatmap.
        affinity_heatmaps (List[str]): Пути к affinity heatmap.
        cache_images (Optional[torch.Tensor]): Кэшированные изображения (если caching=True).
        cache_heatmaps (Optional[torch.Tensor]): Кэшированные тепловые карты (если caching=True).
    """
    
    def __init__(
        self, 
        feature_extractor: Any, 
        dataset: Dict[str, List[str]], 
        caching: bool = False
    ):
        """
        Инициализация датасета.

        Параметры:
            feature_extractor (Any): Объект для предобработки изображений.
            dataset (Dict[str, List[str]]): Словарь с путями к изображениям и heatmaps.
                - "image_path": Пути к изображениям.
                - "character_heatmap_path": Пути к character heatmap.
                - "affinity_heatmaps_path": Пути к affinity heatmap.
            caching (bool, optional): Флаг кэширования данных в память. По умолчанию False.
        """
        self.feature_extractor = feature_extractor
        self.images = dataset['image_path']
        self.character_heatmaps = dataset['character_heatmap_path']
        self.affinity_heatmaps = dataset['affinity_heatmaps_path']

        self.cache_images: Optional[torch.Tensor] = None
        self.cache_heatmaps: Optional[torch.Tensor] = None

        if caching:
            self._cache_images()

    def _prepare_sample(
        self, 
        image: str, 
        character_heatmap: str, 
        affinity_heatmap: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Загружает изображение и его heatmaps, выполняя предобработку.

        Параметры:
            image (str): Путь к изображению.
            character_heatmap (str): Путь к character heatmap.
            affinity_heatmap (str): Путь к affinity heatmap.
        
        Возвращает:
            Tuple[torch.Tensor, torch.Tensor]:
                - Обработанное изображение (3, 512, 512)
                - Heatmap (2, 512, 512)
        """
        image = Image.open(image)
        character_heatmap = Image.open(character_heatmap)
        affinity_heatmap = Image.open(affinity_heatmap)
    
        encoding = self.feature_extractor(
            image,
            size=512,
            do_resize=True,
            do_normalize=True,
            return_tensors="pt"
        )

        image = encoding['pixel_values'].squeeze(0)
        character_heatmap = character_heatmap.resize((512, 512), resample=Image.BICUBIC)
        affinity_heatmap = affinity_heatmap.resize((512, 512), resample=Image.BICUBIC)

        character_heatmap = np.array(character_heatmap, dtype=np.float32) / 255.
        affinity_heatmap = np.array(affinity_heatmap, dtype=np.float32) / 255.

        heatmap = np.stack([character_heatmap, affinity_heatmap], axis=0)
        heatmap = torch.from_numpy(np.clip(heatmap, 0, 1))  
    
        return image, heatmap
    
    def _cache_images(self) -> None:
        """
        Кэширует изображения и heatmaps в память параллельно.
        """
        images, heatmaps = [], []
        
        values = [(a, b, c) for a, b, c in zip(self.images, self.character_heatmaps, self.affinity_heatmaps)]

        # Используем ThreadPoolExecutor для многопоточной обработки
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            results = list(tqdm(
                executor.map(lambda v: self._prepare_sample(*v), values),
                total=len(values),
                desc='Caching...',
                leave=False
            ))

        # Распаковка результатов
        for img, hmap in results:
            images.append(img)
            heatmaps.append(hmap)

        self.cache_images = torch.stack(images)
        self.cache_heatmaps = torch.stack(heatmaps)

        del images, heatmaps
        gc.collect()

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Возвращает элемент датасета по индексу.
        
        Параметры:
            idx (int): Индекс элемента.
        
        Возвращает:
            Dict[str, torch.Tensor]:
                - "pixel_values": Тензор изображения (3, 512, 512)
                - "labels": Тензор heatmaps (2, 512, 512)
        """
        if self.cache_images is not None:
            pixel_values, labels = self.cache_images[idx], self.cache_heatmaps[idx]
        else:
            pixel_values, labels = self._prepare_sample(
                self.images[idx], self.character_heatmaps[idx], self.affinity_heatmaps[idx]
            )
      
        return {
            "pixel_values": pixel_values,  
            "labels": labels               
        }

    def __len__(self) -> int:
        """
        Возвращает количество элементов в датасете.
        """
        return len(self.images)


def collect_data(dataset_folder):
    dataset_folder = os.path.abspath(dataset_folder)

    images_folder = os.path.join(dataset_folder, 'images')
    character_heatmaps_folder = os.path.join(dataset_folder, 'character_heatmaps')
    affinity_heatmaps_folder = os.path.join(dataset_folder, 'affinity_heatmaps')
    
    images = sorted(os.listdir(images_folder))
    character_heatmaps = sorted(os.listdir(character_heatmaps_folder))
    affinity_heatmaps = sorted(os.listdir(affinity_heatmaps_folder))
    
    images = [os.path.join(images_folder, image) for image in images]
    character_heatmaps = [os.path.join(character_heatmaps_folder, heatmap) for heatmap in character_heatmaps]
    affinity_heatmaps = [os.path.join(affinity_heatmaps_folder, heatmap) for heatmap in affinity_heatmaps]

    return pd.DataFrame(
        data=zip(images, character_heatmaps, affinity_heatmaps), 
        columns=['image_path', 'character_heatmap_path', 'affinity_heatmaps_path']
    )


def craft_data_collator(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])  
    labels = torch.stack([item["labels"] for item in batch])  

    return {
        "pixel_values": pixel_values,
        "labels": labels
    }


def box_iou(box1: Tuple[float, float, float, float],
            box2: Tuple[float, float, float, float]) -> float:
    """
    Расчёт IoU двух прямоугольных bbox в формате (xmin, ymin, xmax, ymax).
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Координаты пересечения
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    # Площадь пересечения
    inter_w = max(0.0, inter_xmax - inter_xmin)
    inter_h = max(0.0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    # Площади самих bbox
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    
    # Суммарная площадь (union)
    union_area = area1 + area2 - inter_area
    if union_area == 0:
        return 0.0

    return inter_area / union_area


def match_bboxes(pred_bboxes: List[Tuple[float, float, float, float]],
                 gt_bboxes: List[Tuple[float, float, float, float]],
                 iou_threshold: float = 0.5
                 ) -> Dict[str, float]:
    """
    Грубое жадное сопоставление предсказанных и истинных bbox-ов:
    - Идём по предсказанным bbox-ам,
      - ищем из оставшихся GT-боксов тот, который даёт макс. IoU
      - если IoU >= iou_threshold, считаем его TP и удаляем этот GT-бокс из пула
      - иначе считаем FP
    - Все неиспользованные GT-боксы идут в FN
    Возвращает dict с:
    - tp, fp, fn
    - списком IoU для всех TP
    """
    matched_gt = set()  # чтобы не матчить один и тот же GT-бокс с несколькими предсказаниями
    tp = 0
    fp = 0
    ious_for_tp = []

    for pred_box in pred_bboxes:
        # Ищем GT-бокс с максимальным IoU
        best_iou = 0.0
        best_gt_idx = None
        for i, gt_box in enumerate(gt_bboxes):
            if i in matched_gt:
                continue  # этот уже заматчили
            current_iou = box_iou(pred_box, gt_box)
            if current_iou > best_iou:
                best_iou = current_iou
                best_gt_idx = i
        
        # Проверяем порог IoU
        if best_iou >= iou_threshold:
            tp += 1
            matched_gt.add(best_gt_idx)
            ious_for_tp.append(best_iou)
        else:
            fp += 1

    fn = len(gt_bboxes) - len(matched_gt)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "ious": ious_for_tp
    }


def compute_metrics(eval_pred, tt=0.1, lt=0.1, lwt=0.1):

    """
    Функция для Trainer, которая считает средний IoU и F1-score
    на основе списков предсказанных и истинных bbox.
    eval_pred: кортеж (predictions, labels) — то, что Trainer возвращает как
               выход eval_step-а.
    """
    
    predictions, labels = eval_pred

    predictions= np.array(nn.functional.interpolate(
        torch.tensor(predictions),
        size=[512, 512], # (height, width)
        mode='bilinear',
        align_corners=False
    ))
    
    # print(predictions.shape)#: (batch_size, 2, H, W)
    # print(labels.shape)#: (batch_size, 2, H, W) 

    
    def process_image(score_text, score_link):
        return craft_utils.getDetBoxes(score_text, score_link, tt, lt, lwt)[0]

    def process_image_gt(score_text, score_link):
        return craft_utils.getDetBoxes(score_text, score_link, 0.2, 0.2, 0.2)[0]

    # Параллельная обработка предсказаний
    pred_bboxes_list = Parallel(n_jobs=-1)(
        delayed(process_image)(
            predictions[idx, 0, ...],
            predictions[idx, 1, ...]
        ) 
        for idx in range(len(predictions))
    )
    
    # Параллельная обработка лейблов меток
    gt_bboxes_list = Parallel(n_jobs=-1)(
        delayed(process_image_gt)(
            labels[idx, 0, ...],
            labels[idx, 1, ...]
        ) 
        for idx in range(len(labels))
    )

    return calculate_metric(pred_bboxes_list, gt_bboxes_list, iou_threshold=0.5)


def calculate_metric(pred_bboxes_list, gt_bboxes_list, iou_threshold=0.5):
    total_tp = 0
    total_fp = 0
    total_fn = 0
    all_ious = []

    for pred_bboxes, gt_bboxes in zip(pred_bboxes_list, gt_bboxes_list):
        match_res = match_bboxes(pred_bboxes, gt_bboxes, iou_threshold)
        total_tp += match_res["tp"]
        total_fp += match_res["fp"]
        total_fn += match_res["fn"]
        all_ious.extend(match_res["ious"])
    
    precision = total_tp / (total_tp + total_fp + 1e-9)
    recall    = total_tp / (total_tp + total_fn + 1e-9)
    f1_score  = 2 * precision * recall / (precision + recall + 1e-9)
    mean_iou  = np.mean(all_ious) if all_ious else 0.0
    
    return {
        "mean_iou": float(mean_iou),
        "f1": float(f1_score),
        "precision": float(precision),
        "recall": float(recall)
    }


def inference_sample(model, feature_extractor, image, tt=0.1, lt=0.1, lwt=0.1):
    model.to('cpu')
    img = image.copy()
    tt, lt, lwt = tt * 255., lt * 255., lwt * 255.
    width, height = img.size

    encoding_image = feature_extractor(
        img,
        size=512,
        do_resize=True,
        do_normalize=True,
        return_tensors="pt"
    )

    encoding_image = encoding_image['pixel_values']

    with torch.no_grad():
        outputs = model(pixel_values=encoding_image, labels=None)

    upsampled_logits = nn.functional.interpolate(
        outputs['logits'],
        size=[height, width],  # (height, width)
        mode='bilinear',
        align_corners=False
    )

    to_pill = ToPILImage()

    character_heatmap = to_pill(upsampled_logits[0][0])
    affinity_heatmap = to_pill(upsampled_logits[0][1])

    boxes_pred, _, _ = craft_utils.getDetBoxes(np.array(character_heatmap), np.array(affinity_heatmap), tt, lt, lwt)

    draw = ImageDraw.Draw(img)
    bbox_color = (255, 0, 0)
    bbox_width = 3

    for bbox in boxes_pred:
        draw.rectangle(bbox, outline=bbox_color, width=bbox_width)

    return img, boxes_pred, (character_heatmap, affinity_heatmap)


def save_training_results(trainer, save_dir):
    log_history = trainer.state.log_history
    df = pd.DataFrame(log_history)

    plt.subplots(1, 2, figsize=(16, 5))
    plt.subplot(1, 2, 1)

    train_loss = df[df["loss"].notna()]
    plt.plot(train_loss["step"], train_loss["loss"], label="Train Loss", marker="o")

    val_loss = df[df["eval_loss"].notna()]
    plt.plot(val_loss["step"], val_loss["eval_loss"], label="Validation Loss", marker="o")

    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    for metric in ["eval_mean_iou", "eval_f1", "eval_precision", "eval_recall"]:
        plt.plot(df[df[metric].notna()]['step'], df[metric].dropna(), label=metric, marker="o")

    plt.xlabel("Training Steps")
    plt.ylabel("Metric Value")
    plt.title("Evaluation Metrics Over Time")
    plt.legend()
    plt.grid()

    plt.savefig(os.path.join(save_dir, 'training_plots.png'), bbox_inches='tight', pad_inches=0.1)
    df.to_csv(os.path.join(save_dir, 'training_log.csv'), index=False)


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True