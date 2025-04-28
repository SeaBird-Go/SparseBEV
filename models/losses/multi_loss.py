import torch.nn as nn
from . import OPENOCC_LOSS


@OPENOCC_LOSS.register_module()
class MultiLoss(nn.Module):

    def __init__(self, loss_cfgs):
        super().__init__()
        
        assert isinstance(loss_cfgs, list)
        self.num_losses = len(loss_cfgs)
        
        losses = []
        for loss_cfg in loss_cfgs:
            losses.append(OPENOCC_LOSS.build(loss_cfg))
        self.losses = nn.ModuleList(losses)

    def forward(self, inputs):
        
        loss_dict = {}
        tot_loss = 0.
        for loss_func in self.losses:
            loss = loss_func(inputs)
            tot_loss = tot_loss + loss
            loss_dict.update({
                loss_func.loss_name: loss
            })
        
        return tot_loss, loss_dict