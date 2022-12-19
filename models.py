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
  def __init__(self, N, d, num_heads=8, dropout=0.1):

    super(SPEncoderBlock, self).__init__()

    self.norm_y = nn.LayerNorm(d)
    self.msa = nn.MultiHeadAttention(
      embed_dim=4,
      num_heads=num_heads,
      dropout=dropout,
      batch_first=True
    )

    self.norm_z = nn.LayerNorm(d)
    self.mlp = nn.Sequential(
      nn.Linear(d, d),
      nn.Gelu(),
      nn.Linear(d, d)
    )

  def forward(x, self):
    y = self.msa(self.norm_y(x)) + x
    z = self.mlp(self.norm_z(y)) + y
    return z

class SPAttention(nn.Module):
  # Spatio-Temporal attention
  def __init__(self, N, d, L):

    super(SPAttention, self).__init__()

    self.decoder_blocks = nn.ModuleList([SPEncoderBlock(N, d) for i in range(L)])

    self.mlp_head = nn

  def forward(x, self):

    for block in self.decoder_blocks:
      x = block(x)

    return x



class SPAModel(nn.module):
  def __init__(self, N, t, h, w, d, c, frames, num_classes):
    
    super(SPAModel, self).__init__()

    self.tubelet = TubeletEncoding(t, h, w, d, c)
    self.pos_emb = nn.Parameter(torch.randn(frames / t, h * w, d))
    # cls token = 
    # Embeddings

    self.transformer = SPAttention(N, d, 12)

    self.linear_classifier = nn.Linear(d, num_classes)
        
  def forward(x, self):

    x = self.tubelet(x)
    x = x + self.pos_emb
    # cls + E

    x = self.transfromer(x)

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



