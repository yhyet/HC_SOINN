"""
KAC (Kolmogorov-Arnold Classifier) 分类器实现

基于Kolmogorov-Arnold表示定理，使用径向基函数(RBF)和样条函数实现非线性分类器。
适用于持续学习场景，提供比线性分类器更强的表达能力。

参考实现: KAC项目 (https://github.com/...)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any


class SplineLinear(nn.Linear):
    """
    样条线性层：用于KAC分类器的最终线性变换
    
    Args:
        in_features: 输入特征维度
        out_features: 输出特征维度
        init_scale: 权重初始化缩放因子
    """
    def __init__(self, in_features: int, out_features: int, init_scale: float = 0.1, **kw) -> None:
        self.init_scale = init_scale
        super().__init__(in_features, out_features, bias=False, **kw)

    def reset_parameters(self) -> None:
        """重置参数：根据init_scale进行截断正态分布初始化或零初始化"""
        if self.init_scale == 0:
            nn.init.zeros_(self.weight)
        else:
            nn.init.trunc_normal_(self.weight, mean=0, std=self.init_scale)


class RadialBasisFunction(nn.Module):
    """
    径向基函数(RBF)层：将输入特征映射到高维RBF空间
    
    Args:
        grid_min: 网格最小值
        grid_max: 网格最大值
        num_grids: 网格点数量
        denominator: 平滑度控制参数（None时自动计算）
    """
    def __init__(
        self,
        grid_min: float = -2.,
        grid_max: float = 2.,
        num_grids: int = 8,
        denominator: Optional[float] = None,
    ):
        super().__init__()
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = nn.Parameter(grid, requires_grad=False)
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)

    def forward(self, x):
        """
        前向传播：计算RBF展开
        
        Args:
            x: 输入特征 [B, D]
            
        Returns:
            RBF展开后的特征 [B, D, num_grids]
        """
        return torch.exp(-((x[..., None] - self.grid) / self.denominator) ** 2)


class KACLayer(nn.Module):
    """
    KAC (Kolmogorov-Arnold Classifier) 层
    
    基于Kolmogorov-Arnold表示定理实现的非线性分类器，通过RBF展开和样条线性变换
    提供比传统线性分类器更强的表达能力。
    
    Args:
        input_dim: 输入特征维度（默认768，ViT特征维度）
        output_dim: 输出类别数
        grid_min: RBF网格最小值
        grid_max: RBF网格最大值
        num_grids: RBF网格点数量（默认16）
        use_base_update: 是否使用基更新（当前实现中未使用）
        base_activation: 基激活函数（当前实现中未使用）
        spline_weight_init_scale: 样条权重初始化缩放因子
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        grid_min: float = -2.,
        grid_max: float = 2.,
        num_grids: int = 16,
        use_base_update: bool = False,  # 简化实现，不使用base_update
        base_activation = None,
        spline_weight_init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_grids = num_grids
        
        # LayerNorm归一化
        self.layernorm = nn.LayerNorm(input_dim)
        
        # RBF展开层
        self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)
        
        # 基函数线性层（可选，当前实现中未使用）
        self.basis_linear = nn.Parameter(torch.zeros([input_dim, num_grids]))
        nn.init.trunc_normal_(self.basis_linear, mean=0, std=spline_weight_init_scale)
        
        # 样条线性层：将展开的RBF特征映射到类别空间
        self.spline_linear = SplineLinear(input_dim * num_grids, output_dim, 0)
        
        # 基更新相关（当前实现中未使用）
        self.use_base_update = use_base_update
        if use_base_update:
            self.base_activation = base_activation or F.silu
            self.base_linear = nn.Linear(input_dim, output_dim)
        else:
            self.base_linear = None

    def forward(self, x, time_benchmark: bool = False):
        """
        前向传播
        
        Args:
            x: 输入特征 [B, input_dim]
            time_benchmark: 是否跳过LayerNorm（用于性能测试）
            
        Returns:
            分类logits [B, output_dim]
        """
        # 步骤1: LayerNorm归一化
        if not time_benchmark:
            normalized_x = self.layernorm(x)
        else:
            normalized_x = x
        
        # 步骤2: RBF展开 [B, input_dim] -> [B, input_dim, num_grids]
        spline_basis = self.rbf(normalized_x)
        
        # 步骤3: 展平并线性变换 [B, input_dim, num_grids] -> [B, input_dim*num_grids] -> [B, output_dim]
        flattened = spline_basis.view(*spline_basis.shape[:-2], -1)
        ret = self.spline_linear(flattened)
        
        # 如果使用基更新，添加基线性层的输出（当前实现中未使用）
        if self.use_base_update and self.base_linear is not None:
            base_out = self.base_linear(self.base_activation(normalized_x))
            ret = ret + base_out
        
        return ret

    def update(self, new_output_dim: int):
        """
        增量学习场景下的类别扩展
        
        Args:
            new_output_dim: 新的输出类别数（必须 >= 当前output_dim）
        """
        if new_output_dim < self.output_dim:
            raise ValueError(f"new_output_dim ({new_output_dim}) must be >= current output_dim ({self.output_dim})")
        
        if new_output_dim == self.output_dim:
            return  # 无需更新
        
        # 保存旧权重
        old_weight = self.spline_linear.weight.data.clone()
        
        # 创建新的样条线性层
        new_spline_linear = SplineLinear(
            self.input_dim * self.num_grids, 
            new_output_dim, 
            self.spline_linear.init_scale
        )
        
        # 复制旧权重
        new_spline_linear.weight.data[:self.output_dim, :] = old_weight
        
        # 新类别的权重使用截断正态分布初始化
        if new_output_dim > self.output_dim:
            nn.init.trunc_normal_(
                new_spline_linear.weight.data[self.output_dim:, :],
                mean=0,
                std=self.spline_linear.init_scale
            )
        
        # 替换层
        self.spline_linear = new_spline_linear
        self.output_dim = new_output_dim

    def extra_repr(self) -> str:
        """返回模块的字符串表示"""
        return f'input_dim={self.input_dim}, output_dim={self.output_dim}, num_grids={self.num_grids}'


class KACClassifier(nn.Module):
    """
    KAC分类器包装类：提供与nn.Linear兼容的接口
    
    这个类主要用于向后兼容和接口统一，实际使用时可以直接使用KACLayer。
    """
    def __init__(
        self,
        input_dim: int = 768,
        output_dim: int = 10,
        grid_min: float = -2.,
        grid_max: float = 2.,
        num_grids: int = 16,
        spline_weight_init_scale: float = 0.1,
    ):
        super().__init__()
        self.kac_layer = KACLayer(
            input_dim=input_dim,
            output_dim=output_dim,
            grid_min=grid_min,
            grid_max=grid_max,
            num_grids=num_grids,
            spline_weight_init_scale=spline_weight_init_scale,
        )
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x):
        """前向传播"""
        return self.kac_layer(x)

    def update(self, new_output_dim: int):
        """更新输出维度"""
        self.kac_layer.update(new_output_dim)
        self.output_dim = new_output_dim

    @property
    def out_features(self):
        """返回输出特征数（兼容nn.Linear接口）"""
        return self.output_dim

    @property
    def weight(self):
        """返回权重（兼容nn.Linear接口）"""
        # 注意：KAC的权重结构不同，这里返回样条线性层的权重
        return self.kac_layer.spline_linear.weight





