'''
Copyright (c) 2025 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2025-02-25 11:28:50
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''
import os
import os.path as osp
from tqdm import tqdm
import numpy as np
import pickle
import torch

from mmengine.registry import MODELS
from mmengine.model import BaseModule


@MODELS.register_module()
class InteractModule(BaseModule):
    def __init__(self,
                 **kwargs,
                 ):
        super(self).__init__()

    def forward(self,
                ):
        pass
    
