import torch
import torch.nn as nn
import math
from einops import rearrange, repeat
import functorch

class TubeletEncoding(nn.Module):
  """ Tubelet Encoding.
  Parameters:
    t: length of the tubelets.
    h: height of the tubelets.
    w: width of the tubelets.
    hidden: hidden size.
    c: Number of channels in the input.
  """
  def __init__(self, t, h, w, hidden, c):

    super(TubeletEncoding, self).__init__()
    self.projection = nn.Conv3d(in_channels=c, out_channels=hidden, 
                               kernel_size=(t, h, w),
                               stride=(t, h, w))

  def forward(self, x):
    # Create the tubelets
    x = self.projection(x)
    return rearrange(x, 'b hid t h w -> b (t h w) hid')

class EncoderBlock(nn.Module):
  """ Encoder block.
  Parameters:
    hidden: hidden size.
    num_heads: NUmber of heads in the MultiheadAttention.
  """
  def __init__(self, hidden, drop_layer_rate, block_n, L, num_heads=12):

    super(EncoderBlock, self).__init__()

    self.hidden = hidden
    self.droplayer_p = block_n / max(L - 1, 1) * drop_layer_rate

    self.norm1 = nn.LayerNorm(hidden)
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

  def drop_layer_pattern(self, x):
    if self.training:
      shape = (x.shape[0],) + (1,) * (x.ndim - 1)
      return torch.empty(shape).bernoulli(self.droplayer_p).to(x.device)
    return 0.0

  def forward(self, x):
    y = self.norm1(x)
    x = self.msa(y, y, y)[0] * (1.0 - self.drop_layer_pattern(x)) + x
    
    x = self.mlp(self.norm2(x)) * (1.0 - self.drop_layer_pattern(x)) + x
    return x

class Transformer(nn.Module):
  """ Transformer.
  Parameters:
    hidden: hidden size.
    L: Number of layers.
  """
  def __init__(self, hidden, L, drop_layer_rate):

    super(Transformer, self).__init__()
    self.L = L
    self.decoder_blocks = nn.ModuleList([EncoderBlock(hidden, drop_layer_rate, i, L) for i in range(L)])
    self.encodernorm = nn.LayerNorm(hidden)

  def add_pretrained_weights(self, npz):
    with torch.no_grad():
      # Add weights to the Attention blocks
      for i in range(self.L):
        self.decoder_blocks[i].add_pretrained_weights(npz, i)

      # Encoder norm weights
      self.encodernorm.weight.copy_(torch.from_numpy(npz[f'Transformer/encoder_norm/scale']))
      self.encodernorm.bias.copy_(torch.from_numpy(npz[f'Transformer/encoder_norm/bias']))

  def forward(self, x):

    for block in self.decoder_blocks:
      x = block(x)
    x = self.encodernorm(x)
    return x


class SPAModel(nn.Module):
  """ Model 1 Spatio-temporal attention.
  Parameters:
    frame_size: frame size of the input.
    t: length of the tubelets.
    h: height of the tubelets.
    w: width of the tubelets.
    hidden: hidden size.
    c: Number of channels in the input.
    frames:  NUmber of the sampled frames.
    num_classes: NUmber of classes in the data set.
    L: Number of layers in the transformer.
  """

  def __init__(self, frame_size=224, t=2, h=16, w=16, hidden=768, c=3, frames=32, num_classes=87, L=12, drop_layer_rate=0.3):
    
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
    self.transformer = Transformer(hidden, L, drop_layer_rate)

    self.mlp_head = nn.Linear(hidden, num_classes)
        
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
    # Tubelet embedding
    x = self.tubelet(x)

    # Add classification token
    cls_tokens = self.cls.repeat(x.shape[0], 1, 1)
    x = torch.cat((cls_tokens, x), dim=1)

    # Add positional embedding
    x += self.pos_emb.repeat(x.shape[0], 1, 1, 1).view(x.shape)

    # Attention blocks
    x = self.transformer(x)
    
    # Classify based on the cls tokens
    x = self.mlp_head(x[:, 0, :])
  
    return x



class FactorizedEncoder(nn.Module):
  """ Model 2 Factorized encoder.
  Parameters:
    frame_size: frame size of the input.
    t: length of the tubelets.
    h: height of the tubelets.
    w: width of the tubelets.
    hidden: hidden size.
    c: Number of channels in the input.
    frames:  NUmber of the sampled frames.
    num_classes: NUmber of classes in the data set.
    Ls: Number of layers in the Spatial transformer.
    Lt: Number of layers in the Temporal transformer.
  """


  def __init__(self, frame_size=224, t=2, h=16, w=16, hidden=768, c=3, frames=32, num_classes=87, Ls=12, Lt=4, drop_layer_rate=0.3):

    super(FactorizedEncoder, self).__init__()

    self.nt = frames // t
    self.hidden = hidden
    self.t = t
    nh = frame_size // h
    nw = frame_size // w


    # Embeddings
    self.tubelet = TubeletEncoding(t, h, w, hidden, c)
    self.pos_emb = nn.Parameter(torch.zeros((1, nh * nw + 1, hidden)))

    # CLS tokens
    self.spatial_cls = nn.Parameter(torch.zeros((1, 1, hidden)))
    self.temporal_cls = nn.Parameter(torch.zeros((1, 1, hidden)))

    self.SpatialTransformer = Transformer(hidden, Ls, drop_layer_rate)

    self.TemporalTransformer = Transformer(hidden, Lt, drop_layer_rate)

    self.mlp_head = nn.Linear(hidden, num_classes)

  def add_pretrained_weights(self, npz):
    with torch.no_grad():
      nt = self.nt
      self.pos_emb.copy_(torch.from_numpy(npz['Transformer/posembed_input/pos_embedding']))
      self.spatial_cls.copy_(torch.from_numpy(npz['cls']))
      self.temporal_cls.copy_(torch.from_numpy(npz['cls']))

      # Central frame initialization for embedding weights
      embedding_weights = torch.from_numpy(npz['embedding/kernel'].transpose([3, 2, 0, 1]))  / self.t
      self.tubelet.projection.weight.copy_(repeat(embedding_weights, 'hid c w h -> hid c t w h', t=self.t))
      self.tubelet.projection.bias.copy_(torch.from_numpy(npz['embedding/bias']))

      self.SpatialTransformer.add_pretrained_weights(npz)
      self.TemporalTransformer.add_pretrained_weights(npz)

  def forward(self, x):
    b = x.shape[0]

    # Tubelet embedding
    x = self.tubelet(x)
    x = x.view(x.shape[0], self.nt, -1, self.hidden)
    
    # Add spatial classification token
    spatial_tokens = self.spatial_cls.repeat(x.shape[0], self.nt, 1, 1)
    x = torch.cat((spatial_tokens, x), dim=2)

    # Add positional embedding
    x += self.pos_emb.repeat(self.nt * b, 1, 1).view(x.shape)

    x = x.view(-1, x.shape[2], self.hidden)
    x = self.SpatialTransformer(x)
    x = x.view(b, self.nt, -1, self.hidden)

    # Use spatial cls-token as a input for the temporal transformer
    x = x[:, :, 0, :]

    # Add temporal classification token
    temporal_tokens = self.temporal_cls.repeat(x.shape[0], 1, 1)
    x = torch.cat((temporal_tokens, x), dim=1)

    x = self.TemporalTransformer(x)

    # Classify based on the temporal cls tokens
    x = x[:, 0, :]
    x = self.mlp_head(x)
    return x