import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_dim, head_num, dropout_rate):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.d_k = self.hidden_dim // self.head_num
        
        assert self.hidden_dim == self.d_k * self.head_num

        self.to_q = nn.Linear(hidden_dim, hidden_dim)        
        self.to_k = nn.Linear(hidden_dim, hidden_dim)        
        self.to_v = nn.Linear(hidden_dim, hidden_dim)        
        self.to_o = nn.Linear(hidden_dim, hidden_dim)    
        
        self.dropout_1 = nn.Dropout(dropout_rate)
        self.dropout_2 = nn.Dropout(dropout_rate)
        
        
    def forward(self, x, attn_mask=None):
        batch_size, seq_len, dim_num = x.shape
        
        q = self.to_q(x).reshape(batch_size, seq_len, self.head_num, self.d_k).transpose(1, 2)
        k = self.to_k(x).reshape(batch_size, seq_len, self.head_num, self.d_k).transpose(1, 2)
        v = self.to_v(x).reshape(batch_size, seq_len, self.head_num, self.d_k).transpose(1, 2)
        
        attn_prob = q @ k.transpose(-1, -2) / (self.d_k ** 0.5)
        
        if attn_mask is not None:
            attn_prob = attn_prob.masked_fill(attn_mask==0, -1e9)
            
        attn_prob = F.softmax(attn_prob, dim=-1)
        attn_prob = self.dropout_1(attn_prob)
        
        attn_out = attn_prob @ v
        
        attn_out = attn_out.transpose(1, 2).reshape(batch_size, seq_len, dim_num)
        attn_out = self.to_o(attn_out)
        attn_out = self.dropout_2(attn_out)
        
        return attn_out