import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice для многоканальной (multi-label) бинарной сегментации"""
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        inputs: [B, C, H, W] -- "сырые" логиты или уже вероятности.
        targets: [B, C, H, W] -- бинарная разметка по каждому каналу.
        """
        # Если вход -- логиты, нужно сначала прогнать через сигмоиду
        inputs = torch.sigmoid(inputs)

        # Суммируем по пространству (H,W), остаются размерности [B, C]
        intersection = (inputs * targets).sum(dim=(2,3))
        union = inputs.sum(dim=(2,3)) + targets.sum(dim=(2,3))

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        # dice имеет shape [B, C]. Часто усредняют по batch и по каналам:
        dice_loss = 1.0 - dice  # shape [B, C]

        return dice_loss.mean()  # => скаляр


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: [B, C, H, W] -- логиты (до сигмоиды)
        targets: [B, C, H, W] -- бинарные метки (0 или 1)
        """
        # Превращаем logits в вероятности
        probs = torch.sigmoid(inputs).clamp(1e-4, 1 - 1e-4)

        # pt для каждого канала:
        # pt = p*target + (1-p)*(1-target)
        pt = probs * targets + (1 - probs) * (1 - targets)

        # (1-pt)^gamma
        focal_weight = (1 - pt).pow(self.gamma)

        # alpha_t
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        bce = F.binary_cross_entropy(probs, targets, reduction='none')  # [B, C, H, W]

        loss = alpha_t * focal_weight * bce  # [B, C, H, W]

        if self.reduction == 'mean':
            return loss.mean()   # усредняем по всем B,C,H,W
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss  # будет [B,C,H,W]



class DiceFocalLoss(nn.Module):
    def __init__(self,
                 dice_weight=1.0,
                 focal_weight=1.0,
                 alpha=0.8,
                 gamma=2.0,
                 smooth=1e-6):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice = DiceLoss(smooth=smooth)
        self.focal = FocalLoss(alpha=alpha, gamma=gamma, reduction='mean')

    def forward(self, inputs, targets):
        dice_val = self.dice(inputs, targets)   # скаляр
        focal_val = self.focal(inputs, targets) # скаляр
        loss = self.dice_weight * dice_val + self.focal_weight * focal_val
        return loss