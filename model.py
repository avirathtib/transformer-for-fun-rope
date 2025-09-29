import math
import token

import torch
import torch.nn as nn
from torch.nn import functional as F

class GeLU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, dim) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)

    def forward(self, x):
        x = self.embedding(x)
        return x
class RoPE():
    def __init__(self, embed_dim, seq_length, device, theta = 10000):
        dims = torch.arange(0, embed_dim, 2, device = device)
        thetas = 1 / ((theta) ** (dims / embed_dim))
        toks = torch.arange(seq_length, device = device)
        angles = torch.outer(toks, thetas)
        self.angles_rotated = torch.polar(torch.ones_like(angles), angles)

    def __call__(self):
        return self.angles_rotated

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, qkv_embed_dim, n_heads, dropout):
        super().__init__()            
        self.to_qkv = nn.Linear(embed_dim, qkv_embed_dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.n_heads = n_heads
        self.head_dim = embed_dim // self.n_heads
        self.qkv_embed_dim = qkv_embed_dim
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj.NANOGPT_SCALE_INIT = True




    def forward(self, x): 
        # batch size will be (B, S, D)
        # print(x.shape)
        B, S, D = x.shape
        self.mask = torch.tril(torch.ones(x.shape[1], x.shape[1], device = x.device))
        qkv = self.to_qkv(x) # B, S, QKV*3
        q, k, v = qkv.chunk(3, dim=-1)
        # print("shapes here", q.shape, k.shape, v.shape)
        q = q.view(q.shape[0], q.shape[1], self.n_heads, q.shape[2] // self.n_heads).transpose(1, 2)
        v = v.view(v.shape[0], v.shape[1], self.n_heads, v.shape[2] // self.n_heads).transpose(1, 2)
        k = k.view(k.shape[0], k.shape[1], self.n_heads, k.shape[2] // self.n_heads).transpose(1, 2)
        self.qrope = RoPE(self.head_dim, x.shape[1], x.device)
        self.krope = RoPE(self.head_dim, x.shape[1], x.device)
        # self.vrope = RoPE(self.head_dim, x.shape[1], x.device)
        q_dim_by_two = q.view(q.shape[0], q.shape[1], q.shape[2], q.shape[3] // 2 , 2) #B, N, S, D // 2, 2
        q_complex = torch.view_as_complex(q_dim_by_two)
        # print("qshape", q_dim_by_two.shape)
        # print("q_complex", q_complex.shape)
        q_angles = self.qrope() #s, d//2
        # print("qangles", q_angles.shape)
        q_angles = q_angles.unsqueeze(0).unsqueeze(0)
        q_rotated = q_complex * q_angles
        q_with_pos = torch.view_as_real(q_rotated).reshape(q_rotated.shape[0], q_rotated.shape[1], q_rotated.shape[2], q_rotated.shape[3] * 2)
        k_dim_by_two = k.view(k.shape[0], k.shape[1], k.shape[2], k.shape[3] // 2 , 2) #B, N, S, D // 2, 2
        k_complex = torch.view_as_complex(k_dim_by_two)
        k_angles = self.krope() #s, d//2
        k_angles = k_angles.unsqueeze(0).unsqueeze(0)
        k_rotated = k_complex * k_angles
        k_with_pos = torch.view_as_real(k_rotated).reshape(k_rotated.shape[0], k_rotated.shape[1], k_rotated.shape[2], k_rotated.shape[3] * 2)
        attention = (q_with_pos @ k_with_pos.transpose(-1, -2)) / (self.head_dim ** 0.5) # B, N, S, D @ B, N, D, S = B, N, S, S
        unsqueezed_mask = self.mask.unsqueeze(0).unsqueeze(0)
        attention_masked = attention.masked_fill(unsqueezed_mask == 0, float('-inf'))
        attention_softmax = torch.nn.functional.softmax(attention_masked, dim = -1)
        # print("Attention pattern (first head, first sample):")
        # print(attention_softmax[0, 0, :5, :5])  # Should be lower triangular
        # print(attention_softmax.shape)
        # attention_softmax = self.attention_dropout(attention_softmax)
        # print("here",attention_softmax.shape)
        # print("herehere",v.shape)
        final_scores = attention_softmax @ v # B N S S * B N S D =
        # print(v.shape)
        # print(final_scores.shape)
        final_scores = final_scores.transpose(1, 2).contiguous().view(B, S, D)
        out = self.out_proj(final_scores)
        # out  = out.view(out.shape[0], out.shape[2], out.shape[1] * out.shape[3])
        return out

class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = GeLU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.fc2.NANOGPT_SCALE_INIT = True
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return (self.fc2(self.act(self.fc1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, qkv_embed_dim, n_heads, dropout):
        super().__init__()
        self.mha = MultiHeadAttention(embed_dim, qkv_embed_dim, n_heads, dropout)
        self.ffn = FeedForward(embed_dim, embed_dim * 4, dropout)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, qkv_dim, n_layers, n_heads, dropout):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, embed_dim)
        self.layers = nn.Sequential(*[TransformerBlock(embed_dim, qkv_dim, n_heads, dropout) for n in range(n_layers)])
        self.norm = nn.LayerNorm(embed_dim)        
        self.fc = nn.Linear(embed_dim, vocab_size)
        self.fc.weight = self.token_embedding.embedding.weight
        self.fc.bias = None
        self.vocab_size = vocab_size

    def forward(self, x):
        B, S = x.shape[0], x.shape[1]
        token_embedding = self.token_embedding(x)
        out = self.layers(token_embedding)
        norm_out = self.norm(out)
        logits = self.fc(norm_out)
        # logits = logits.reshape(B * S, self.vocab_size) 
        return logits
           


        