import torch
from einops import rearrange
from einops.layers.torch import Rearrange

class TubeletEncoding(nn.module):
  def __init__(self, t, h, w, d, c):

    super(TubeletEncoding, self).__init__()
    tubelet_dim = c * t * h * w
    self.embedding = n.Sequential(
      Rearrange('b (t t1) c (h h1) (w w1) <- b t (h w) (t2 h1 w1 c)', t1=t, h1=h, w1=w),
      nn.Linear(tubelet_dim, d)
    )

  def forward(self, x):
    return self.embedding(x)
    

class SPEncoderBlock(nn.module):
  # Encoder block for the Spatio-Temporal attention
  def __init__(self, N, d, hidden, num_heads=8):

    super(SPEncoderBlock, self).__init__()

    self.hidden = hidden

    self.norm1 = nn.LayerNorm(d)
    self.msa = nn.MultiHeadAttention(
      embed_dim=4,
      num_heads=num_heads,
      batch_first=True
    )

    self.norm2 = nn.LayerNorm(d)
    self.mlp = nn.Sequential(
      nn.Linear(d, d),
      nn.Gelu(),
      nn.Linear(d, d)
    )

  def add_pretrained_weights(npz, block_n, self):
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

      self.msa.out_proj_weight.copy_(torch.from_numpy(npz[f'{BLOCK}/{ATT_OUT}/kernel']).view(self.hidden, self.hidden)
  )

      # MSA biases
      k_bias = torch.from_numpy(npz[f'{BLOCK}/{ATT_K}/bias']).view(-1)
      q_bias = torch.from_numpy(npz[f'{BLOCK}/{ATT_Q}/bias']).view(-1)
      v_bias = torch.from_numpy(npz[f'{BLOCK}/{ATT_V}/bias']).view(-1)

      self.msa.in_proj_bias.copy_(torch.cat((k_bias, q_bias, v_bias)))

      self.msa.out_proj_bias.copy_(torch.from_numpy(npz[f'{BLOCK}/{ATT_OUT}/bias']))

      # MLP weights
      self.mlp[0].weight.copy(torch.from_numpy(npz[f'{BLOCK}/{MLP_1}/scale']))
      self.mlp[0].bias.copy(torch.from_numpy(npz[f'{BLOCK}/{MLP_1}/bias']))
      self.mlp[2].weight.copy(torch.from_numpy(npz[f'{BLOCK}/{MLP_1}/scale']))
      self.mlp[2].bias.copy(torch.from_numpy(npz[f'{BLOCK}/{MLP_1}/bias']))

  def forward(x, self):
    y = self.msa(self.norm1(x)) + x
    z = self.mlp(self.norm2(y)) + y
    return z

class SPAttention(nn.Module):
  # Spatio-Temporal attention
  def __init__(self, N, d, L):

    super(SPAttention, self).__init__()

    self.decoder_blocks = nn.ModuleList([SPEncoderBlock(N, d) for i in range(L)])

  def add_pretrained_weights(npz, self):
    with torch.no_grad():
      # Add weights to the Attention blocks
      for i in range(12):
        self.decoder_blocks[i].add_pretrained_weights(npz, i)

      # 

      

  def forward(x, self):

    for block in self.decoder_blocks:
      x = block(x)

    return x


class SPAModel(nn.module):
  def __init__(self, N, t, h, w, d, hidden, c, frames, num_classes):
    
    super(SPAModel, self).__init__()
    self.nt = frames / t
    self.tubelet = TubeletEncoding(t, h, w, d, c)
    self.pos_emb = nn.Parameter(torch.zeros(self.nt, h * w + 1, hidden))
    self.cls = nn.Parameter(torch.zeros(1, 1, hidden))
    self.patch_emb = nn.Parameter(torch.zeros(self.nt, h * w + 1, hidden))
    # Embeddings

    self.transformer = SPAttention(N, d, 12)

    self.linear_classifier = nn.Linear(hidden, num_classes)
        
  def add_pretrained_weights(npz, self):
    with torch.no_grad():

      self.pos_emb = npz['Transformer/posembed_input/pos_embedding'].repeat(self.nt, 1, 1)
      self.cls = npz['cls']
      self.patch_emb[]

  def forward(x, self):

    x = self.tubelet(x)
    x = x + self.pos_emb
    # cls + E

    x = self.transformer(x)

    x = linear_classifier(x[:, 0])

    return x



class FactorizedEncoder(nn.module):
  def __init__(self, N, t, h, w, d, c, frames, num_classes):

    super(FactorizedEncoder, self).__init__()

    self.tubelet = TubeletEncoding(t, h, w, d, c)
    self.pos_emb = nn.Parameter(torch.randn(frames / t, h * w, d))
    # cls token = 
    # Embeddings

    self.SpatialTransformer = SPAttention(N, d, 12)
    self.spatialClassifier = nn.Linear(d, num_classes)

    self.TemporalTransformer = SPAttention(N, d, 12)
    self.temporalClassifier = nn.Linear(d, num_classes)

  def forward(x, self)



