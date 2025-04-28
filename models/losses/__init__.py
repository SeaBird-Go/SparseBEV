from mmengine.registry import Registry
OPENOCC_LOSS = Registry('openocc_loss')

from .multi_loss import MultiLoss
from .photometric_loss import PhotometricLoss
from .cd_loss import ChamferDistanceLoss
