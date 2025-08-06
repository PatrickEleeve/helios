# helios/adapters/adapter.py

import torch
import torch.nn as nn

class SemanticAdapter(nn.Module):
    """
    一个可学习的语义适配器，作为连接不同Agent隐藏状态的“神经链路”。
    它的作用是将一个源语义空间中的隐藏状态，翻译成目标语义空间中的状态。
    """
    def __init__(self, hidden_size, bottleneck_dim, use_residual=True):
        super().__init__()
        self.use_residual = use_residual
        self.norm = nn.LayerNorm(hidden_size)
        self.down_project = nn.Linear(hidden_size, bottleneck_dim, bias=False)
        self.activation = nn.GELU()
        self.up_project = nn.Linear(bottleneck_dim, hidden_size, bias=False)
        
        # 使用 Xavier 初始化以获得更好的初始性能
        nn.init.xavier_uniform_(self.down_project.weight)
        # 技巧：零初始化up_project，使适配器在训练初期接近于恒等变换，让学习更稳定
        nn.init.zeros_(self.up_project.weight)

    def forward(self, hidden_state):
        residual = hidden_state
        x = self.norm(hidden_state)
        
        # 修正原有的逻辑顺序，应该是 norm -> down_project -> activation -> up_project
        x = self.down_project(x) 
        x = self.activation(x)
        x = self.up_project(x)
        
        if self.use_residual:
            # --- 核心修复：在这里将 x 的数据类型转换为与 residual 一致 ---
            # 这样无论输入是 float16还是float32，加法都能安全执行
            return x.to(residual.dtype) + residual
        else:
            return x