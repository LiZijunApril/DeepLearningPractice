import math
from numpy import add
import torch
from torch import nn, Tensor


# %% 基于位置的前馈网络
class PositionWiseFFN(nn.Module):
    """Position-wise Feed-Forward Network"""

    def __init__(self, ffn_num_input: int, ffn_num_hiddens: int, ffn_num_outputs: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)
        
    def forward(self, X: Tensor):
        return self.dense2(self.relu(self.dense1(X)))

ffn = PositionWiseFFN(4, 4, 8)
ffn.eval()
ffn(torch.ones((2, 3, 4)))[0]
    
# %%
ln = nn.LayerNorm(2)
bn = nn.BatchNorm1d(2)
X = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
print('layer norm:', ln(X), '\nbatch norm:', bn(X))

# %%
class AddNorm(nn.Module):
    """残差连接，followed by layer normalization"""
    def __init__(self, normalized_shape: int, dropout: float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)
        
    def forward(self, X: Tensor, Y: Tensor) -> Tensor:
        return self.ln(self.dropout(Y) + X)

add_norm = AddNorm([3, 4], 0.5) # Normalized_shape is input.size()[1:]
add_norm.eval()