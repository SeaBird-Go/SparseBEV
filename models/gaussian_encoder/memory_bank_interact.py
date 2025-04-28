'''
Copyright (c) 2025 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2025-03-11 16:19:09
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''
import torch, torch.nn as nn
import torch.nn.functional as F
from mmengine.registry import MODELS
from mmengine.model import BaseModule


@MODELS.register_module()
class MemoryBank(BaseModule):
    """Memory bank for each sample.
    """ 
    def __init__(self, feature_dim, capacity, init_cfg=None, **kwargs):  
        """  
        初始化 Memory Bank  
        :param feature_dim: 每个特征的维度  
        :param capacity: Memory Bank 的最大容量  
        """ 
        super().__init__(init_cfg)
        self.feature_dim = feature_dim  
        self.capacity = capacity  
        self.bank = nn.Parameter(
            torch.randn(capacity, feature_dim) * 0.01,
            requires_grad=False)

    def update(self, features, alpha=0.5):  
        """  
        更新 Memory Bank  
        :param features: 新的 query 特征，形状为 [batch_size, feature_dim]  
        """  
        batch_size, feature_dim = features.shape

        # 计算新特征与 Memory Bank 中所有特征的注意力权重  
        for i in range(batch_size):  
            new_feature = features[i]  # [feature_dim]  
            memory_flat = self.bank  # [capacity, feature_dim]  

            # 注意力计算  
            attention_scores = torch.matmul(memory_flat, new_feature)  # [capacity]  
            attention_weights = F.softmax(attention_scores, dim=0)  # [capacity]  

            # 加权更新 Memory Bank  
            self.bank.data = (1 - alpha) * memory_flat + alpha * torch.matmul(attention_weights, memory_flat)  # [capacity, feature_dim]  

    def get_memory(self):  
        """  
        获取 Memory Bank 中的所有特征  
        """  
        return self.bank 
    

@MODELS.register_module()
class MemoryBankV2(BaseModule):
    """Using the memory bank for each query.

    Args:
        BaseModule (_type_): _description_
    """
    def __init__(self, num_queries, feature_dim, capacity):  
        """  
        初始化 Memory Bank  
        :param num_queries: 每个样本的 query 数量  
        :param feature_dim: 每个 query 的特征维度  
        :param capacity: Memory Bank 的最大容量（样本数量）  
        """  
        self.num_queries = num_queries  
        self.feature_dim = feature_dim  
        self.capacity = capacity  
        self.bank = torch.zeros(capacity, num_queries, feature_dim)  # 初始化为零  

    def update(self, query_features):  
        """  
        更新 Memory Bank  
        :param query_features: 新的 query 特征，形状为 [batch_size, num_queries, feature_dim]  
        """  
        batch_size = query_features.size(0)  

        # 计算新特征与 Memory Bank 中所有特征的注意力权重  
        for i in range(batch_size):  
            new_feature = query_features[i]  # [num_queries, feature_dim]  
            memory_flat = self.bank.view(-1, self.feature_dim)  # [capacity * num_queries, feature_dim]  

            # 注意力计算  
            attention_scores = torch.matmul(new_feature, memory_flat.t())  # [num_queries, capacity * num_queries]  
            attention_weights = F.softmax(attention_scores, dim=-1)  # [num_queries, capacity * num_queries]  

            # 加权更新 Memory Bank  
            memory_flat = memory_flat + torch.matmul(attention_weights.t(), new_feature)  # [capacity * num_queries, feature_dim]  
            self.bank = memory_flat.view(self.capacity, self.num_queries, self.feature_dim)  # 恢复形状  

    def get_memory(self):  
        """  
        获取 Memory Bank 中的所有特征  
        """  
        return self.bank
    

@MODELS.register_module()
class MemoryBankInteraction(BaseModule):  
    def __init__(self, feature_dim):  
        """  
        Memory Bank 与 Query 的交互模块  
        :param feature_dim: 特征维度  
        """  
        super(MemoryBankInteraction, self).__init__()  
        self.query_proj = nn.Linear(feature_dim, feature_dim)  
        self.key_proj = nn.Linear(feature_dim, feature_dim)  
        self.value_proj = nn.Linear(feature_dim, feature_dim)  
        self.scale = feature_dim ** 0.5  # 缩放因子  

    def forward(self, queries, memory_bank_features):  
        """  
        前向传播  
        :param queries: Finetune 阶段的 query 特征，形状为 [batch_size, num_queries, feature_dim]  
        :param memory_bank_features: Memory Bank 中的特征，形状为 [memory_size, feature_dim]  
        """  
        # 线性变换  
        Q = self.query_proj(queries)  # [batch_size, num_queries, feature_dim]  
        K = self.key_proj(memory_bank_features)  # [memory_size, feature_dim]  
        V = self.value_proj(memory_bank_features)  # [memory_size, feature_dim]  

        # 注意力计算  
        attention_scores = torch.matmul(Q, K.t()) / self.scale  # [batch_size, num_queries, memory_size]  
        attention_weights = F.softmax(attention_scores, dim=-1)  # [batch_size, num_queries, memory_size]  

        # 加权求和  
        memory_interaction = torch.matmul(attention_weights, V)  # [batch_size, num_queries, feature_dim]  

        return memory_interaction