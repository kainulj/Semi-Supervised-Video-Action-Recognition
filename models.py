import torch
import torch.nn as nn
import math
from einops import rearrange, einsum
from einops.layers.torch import Rearrange
import functorch

class TubeletEncoding(nn.Module):
  def __init__(self, t, h, w, hidden, c):

    super(TubeletEncoding, self).__init__()
    tubelet_dim = c * t * h * w
    self.hidden = hidden
    self.projection = nn.Conv3d(in_channels=c, out_channels=hidden, 
                               kernel_size=(t, h, w),
                               stride=(t, h, w))

  def forward(self, x):
    x = self.projection(x)
    return rearrange(x, 'b hid t h w -> b (t h w) hid')

class SPEncoderBlock(nn.Module):
  # Encoder block for the Spatio-Temporal attention
  def __init__(self, hidden, num_heads=12):

    super(SPEncoderBlock, self).__init__()

    self.hidden = hidden

    self.norm1 = nn.LayerNorm(hidden)
    #self.msa = MSA(hidden)
    self.msa = nn.MultiheadAttention(
      embed_dim=hidden,
      num_heads=num_heads,
      batch_first=True
    )

    self.norm2 = nn.LayerNorm(hidden)
    self.mlp = nn.Sequential(
      nn.Linear(hidden, 4 * hidden),
      nn.GELU(),
      nn.Linear(4 * hidden, hidden)
    )

  def add_pretrained_weights(self, npz, block_n):
    with torch.no_grad():
      ATT_K = "MultiHeadDotProductAttention_1/key"
      ATT_OUT = "MultiHeadDotProductAttention_1/out"
      ATT_Q = "MultiHeadDotProductAttention_1/query"
      ATT_V = "MultiHeadDotProductAttention_1/value"
      ATT_NORM = "LayerNorm_0"
      MLP_NORM = "LayerNorm_2"
      MLP_1 = "MlpBlock_3/Dense_0"
      MLP_2 = "MlpBlock_3/Dense_1"

      BLOCK = f'Transformer/encoderblock_{block_n}'

      # Layernorm weights
      self.norm1.weight.copy_(torch.from_numpy(npz[f'{BLOCK}/{ATT_NORM}/scale']))
      self.norm1.bias.copy_(torch.from_numpy(npz[f'{BLOCK}/{ATT_NORM}/bias']))
      self.norm2.weight.copy_(torch.from_numpy(npz[f'{BLOCK}/{MLP_NORM}/scale']))
      self.norm2.bias.copy_(torch.from_numpy(npz[f'{BLOCK}/{MLP_NORM}/bias']))

      # MSA weights
      k_weight = torch.from_numpy(npz[f'{BLOCK}/{ATT_K}/kernel']).view(self.hidden, self.hidden)
      q_weight = torch.from_numpy(npz[f'{BLOCK}/{ATT_Q}/kernel']).view(self.hidden, self.hidden)
      v_weight = torch.from_numpy(npz[f'{BLOCK}/{ATT_V}/kernel']).view(self.hidden, self.hidden)

      self.msa.in_proj_weight.copy_(torch.cat((k_weight, q_weight, v_weight), 0))

      self.msa.out_proj.weight.copy_(torch.from_numpy(npz[f'{BLOCK}/{ATT_OUT}/kernel']).view(self.hidden, self.hidden)
  )

      # MSA biases
      k_bias = torch.from_numpy(npz[f'{BLOCK}/{ATT_K}/bias']).view(-1)
      q_bias = torch.from_numpy(npz[f'{BLOCK}/{ATT_Q}/bias']).view(-1)
      v_bias = torch.from_numpy(npz[f'{BLOCK}/{ATT_V}/bias']).view(-1)

      self.msa.in_proj_bias.copy_(torch.cat((k_bias, q_bias, v_bias)))

      self.msa.out_proj.bias.copy_(torch.from_numpy(npz[f'{BLOCK}/{ATT_OUT}/bias']))
      # MLP weights
      self.mlp[0].weight.copy_(torch.from_numpy(npz[f'{BLOCK}/{MLP_1}/kernel']).t())
      self.mlp[0].bias.copy_(torch.from_numpy(npz[f'{BLOCK}/{MLP_1}/bias']))
      self.mlp[2].weight.copy_(torch.from_numpy(npz[f'{BLOCK}/{MLP_2}/kernel']).t())
      self.mlp[2].bias.copy_(torch.from_numpy(npz[f'{BLOCK}/{MLP_2}/bias']))

  def forward(self, x):
    x = self.msa(self.norm1(x), self.norm1(x), self.norm1(x))[0] + x
    
    x = self.mlp(self.norm2(x)) + x
    return x

class SPAttention(nn.Module):
  # Spatio-Temporal attention
  def __init__(self, hidden, L):

    super(SPAttention, self).__init__()
    self.L = L
    self.decoder_blocks = nn.ModuleList([SPEncoderBlock(hidden) for i in range(L)])

  def add_pretrained_weights(self, npz):
    with torch.no_grad():
      # Add weights to the Attention blocks
      for i in range(self.L):
        self.decoder_blocks[i].add_pretrained_weights(npz, i)

  def forward(self, x):

    for block in self.decoder_blocks:
      x = block(x)
      print(torch.cuda.memory_allocated() // 10**6)

    return x


class SPAModel(nn.Module):
  def __init__(self, frame_size=224, t=2, h=16, w=16, hidden=768, c=3, frames=32, num_classes=87):
    
    super(SPAModel, self).__init__()
    self.nt = frames // t
    self.hidden = hidden
    self.t = t
    nh = frame_size // h
    nw = frame_size // w

    # Embeddings
    self.tubelet = TubeletEncoding(t, h, w, hidden, c)
    self.pos_emb = nn.Parameter(torch.zeros((1, self.nt * nh * nw + 1, hidden)))
    self.cls = nn.Parameter(torch.zeros((1, 1, hidden)))
    
    # Transformer
    self.transformer = SPAttention(hidden, 2)

    self.linear_classifier = nn.Linear(hidden, num_classes)
        
  def add_pretrained_weights(self, npz):
    with torch.no_grad():
      nt = self.nt
      pos_emb = torch.from_numpy(npz['Transformer/posembed_input/pos_embedding'])
      pos_emb_without_cls = pos_emb[:, 1:, :].repeat(nt, 1, 1).view(1, -1, self.hidden)
      self.pos_emb.copy_(torch.cat((pos_emb[:, 0, :].view(1, 1, -1), pos_emb_without_cls), dim=1))
      self.cls.copy_(torch.from_numpy(npz['cls']))

      # Central frame initialization for embedding weights
      self.tubelet.projection.weight.copy_(torch.zeros_like(self.tubelet.projection.weight))
      self.tubelet.projection.weight[:, :, self.t // 2, :, :].copy_(torch.from_numpy(npz['embedding/kernel'].transpose([3, 2, 0, 1])))
      self.tubelet.projection.bias.copy_(torch.from_numpy(npz['embedding/bias']))

      self.transformer.add_pretrained_weights(npz)

  def forward(self, x):
    x = self.tubelet(x)

    cls_tokens = self.cls.repeat(x.shape[0], 1, 1)
    x = torch.cat((cls_tokens, x), dim=1) 
    x += self.pos_emb.repeat(x.shape[0], 1, 1, 1).view(x.shape)

    x = self.transformer(x)
    x = self.linear_classifier(x[:, 0, :])
    return x



class FactorizedEncoder(nn.Module):
  def __init__(self, frame_size=224, t=2, h=16, w=16, hidden=768, c=3, frames=32, num_classes=87):

    super(FactorizedEncoder, self).__init__()

    self.nt = frames // t
    self.hidden = hidden
    self.t = t
    nh = frame_size // h
    nw = frame_size // w


    # Embeddings
    self.tubelet = TubeletEncoding(t, h, w, hidden, c)
    self.pos_emb = nn.Parameter(torch.zeros((1, nh * nw + 1, hidden)))
    self.spatial_cls = nn.Parameter(torch.zeros((1, 1, hidden)))
    self.temporal_cls = nn.Parameter(torch.zeros((1, 1, hidden)))

    self.SpatialTransformer = SPAttention(hidden, 12)

    self.TemporalTransformer = SPAttention(hidden, 4)
    self.temporalClassifier = nn.Linear(hidden, num_classes)


  def add_pretrained_weights(self, npz):
    with torch.no_grad():
      nt = self.nt
      self.pos_emb.copy_(torch.from_numpy(npz['Transformer/posembed_input/pos_embedding']))
      self.spatial_cls.copy_(torch.from_numpy(npz['cls']))
      self.temporal_cls.copy_(torch.from_numpy(npz['cls']))

      # Central frame initialization for embedding weights
      self.tubelet.projection.weight.copy_(torch.zeros_like(self.tubelet.projection.weight))
      self.tubelet.projection.weight[:, :, self.t // 2, :, :].copy_(torch.from_numpy(npz['embedding/kernel'].transpose([3, 2, 0, 1])))
      self.tubelet.projection.bias.copy_(torch.from_numpy(npz['embedding/bias']))

      self.SpatialTransformer.add_pretrained_weights(npz)
      self.TemporalTransformer.add_pretrained_weights(npz)

  def forward(self, x):
    b = x.shape[0]
    x = self.tubelet(x)
    x = x.view(x.shape[0], self.nt, -1, self.hidden)
    
    spatial_tokens = self.spatial_cls.repeat(x.shape[0], self.nt, 1, 1)
    x = torch.cat((spatial_tokens, x), dim=2)
    x += self.pos_emb.repeat(self.nt * b, 1, 1).view(x.shape)
    x = x.view(-1, x.shape[2], self.hidden)
    x = self.SpatialTransformer(x)
    x = x.view(b, self.nt, -1, self.hidden)
    x = x[:, :, 0, :]

    temporal_tokens = self.temporal_cls.repeat(x.shape[0], 1, 1)
    x = torch.cat((temporal_tokens, x), dim=1)
    x = self.TemporalTransformer(x)
    return self.temporalClassifier(x)




