import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
from mmdet.registry import MODELS



@MODELS.register_module()
class SIRLOSS(nn.Module):
    """Scale-Invariant Logarithmic (SiLog) Loss.

    Args:
        bsize (int): Batch size.
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of the loss.
        avg_non_ignore (bool, optional): If True, only consider non-ignored elements for averaging. Defaults to False.
    """

    def __init__(self,
                 ignore_index = 255,
                 loss_weight: float = 1.0,
                 smooth: bool = False,
                 epsilon:  float=0.1,
                 log:bool=False) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.loss_weight = loss_weight
        self.smooth = smooth
        self.epsilon = epsilon
        self.log = log

    def forward(self,
                pred: Tensor,
                label: Tensor,
                ignore_index: int = 255
                        ) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): The prediction.
            label (Tensor): The target tensor.
            mask (Tensor): The mask tensor.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.

        Returns:
            Tensor: Calculated loss.
        """
        
        # The default value of ignore_index is the same as in `F.cross_entropy`
        ignore_index = -100 if ignore_index is None else ignore_index

        # Mask out ignored elements
        valid_mask = ((label >= 0) & (label != ignore_index)).float()
        
        if self.log:
            # Ensure pred_valid and label_valid are positive before taking log
            pred = torch.clamp(pred, min=1e-6)
            label = torch.clamp(label, min=1e-6)
            pred = torch.log(pred)
            label = torch.log(label)
        if self.smooth:
            label = (1 - self.epsilon) * label + self.epsilon * pred
            
        pred_valid = pred * valid_mask
        label_valid = label * valid_mask            

        diff = (pred_valid - label_valid)
        
        nvalid_pix=torch.sum(valid_mask)

        depth_cost = (torch.sum(nvalid_pix * torch.sum(diff**2))
                      - 0.5 * torch.sum(torch.sum(diff)**2)) \
                     / torch.maximum(torch.sum(nvalid_pix**2), torch.tensor(1.0, device=label.device))


        return self.loss_weight * depth_cost
