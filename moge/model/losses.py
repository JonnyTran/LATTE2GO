import codecs as cs
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_sparse import SparseTensor


class ClassificationLoss(nn.Module):
    def __init__(self, loss_type: str, n_classes: int = None, class_weight: Tensor = None, pos_weight: Tensor = None,
                 multilabel: bool = False, reduction: str = "mean"):
        super().__init__()
        self.n_classes = n_classes
        self.loss_type = loss_type
        self.multilabel = multilabel
        self.reduction = reduction

        print(f"INFO: Using {loss_type}")
        print(f"class_weight for {class_weight.shape} classes") if class_weight is not None else None
        print(f"pos_weight for {pos_weight.shape} classes") if pos_weight is not None else None

        if loss_type == "SOFTMAX_CROSS_ENTROPY":
            self.criterion = torch.nn.CrossEntropyLoss(weight=class_weight, reduction=reduction)
        elif loss_type == "NEGATIVE_LOG_LIKELIHOOD":
            self.criterion = torch.nn.NLLLoss(class_weight, reduction=reduction)
        elif loss_type == "SOFTMAX_FOCAL_CROSS_ENTROPY":
            self.criterion = FocalLoss(n_classes, "SOFTMAX")
        elif loss_type == "SIGMOID_FOCAL_CROSS_ENTROPY":
            self.criterion = FocalLoss(n_classes, "SIGMOID")
        elif loss_type == "BCE_WITH_LOGITS":
            self.criterion = torch.nn.BCEWithLogitsLoss(weight=class_weight, reduction=reduction, pos_weight=pos_weight)
        elif loss_type == "BCE":
            self.criterion = torch.nn.BCELoss(weight=class_weight, reduction=reduction)
        elif loss_type == "MULTI_LABEL_MARGIN":
            self.criterion = torch.nn.MultiLabelMarginLoss(weight=class_weight, reduction=reduction)
        elif loss_type == "KL_DIVERGENCE":
            self.criterion = torch.nn.KLDivLoss(reduction=reduction)
        elif loss_type == "PU_LOSS_WITH_LOGITS":
            self.criterion = PULoss(prior=torch.tensor(1 / 1000))
        elif loss_type == "LINK_PRED_WITH_LOGITS":
            self.criterion = LinkPredLoss()
        elif "CONTRASTIVE" in loss_type:
            assert "LOGITS" not in loss_type
            self.criterion = ContrastiveLoss()
        else:
            raise TypeError(f"Unsupported loss type: {loss_type}")

    def forward(self, logits: Tensor, targets: Tensor, weights: Optional[Tensor] = None) -> Tensor:
        """

        Args:
            logits (torch.Tensor): predicted labels or logits
            targets (torch.Tensor): true labels
            weights (): Sample weights.

        Returns:

        """
        if isinstance(targets, SparseTensor):
            targets = targets.to_dense()

        if self.multilabel:
            assert self.loss_type in ["BCE_WITH_LOGITS", "BCE", "PU_LOSS_WITH_LOGITS",
                                      "SIGMOID_FOCAL_CROSS_ENTROPY", "MULTI_LABEL_MARGIN"], \
                f"Multilabel loss in compatible with loss type: {self.loss_type}"
            targets = targets.type_as(logits)
        else:
            if self.loss_type in ["SOFTMAX_CROSS_ENTROPY", "NEGATIVE_LOG_LIKELIHOOD", "SOFTMAX_FOCAL_CROSS_ENTROPY"] \
                    and targets.dim() == 1:
                targets = torch.eye(self.n_classes, device=logits.device, dtype=torch.long)[targets]

        loss = self.criterion.forward(logits, targets)

        if isinstance(weights, Tensor) and weights.numel() and self.reduction == "none":
            if loss.dim() > 1 and loss.size(1) > 1:
                loss = loss.sum(dim=1)
            loss = (weights * loss).sum() / weights.sum()

        return loss


