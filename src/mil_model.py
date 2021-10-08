import copy
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import RobertaModel as WordEncoder


class MIL(nn.Module):

  def __init__(self, args):
    super(MIL, self).__init__()
    self.args = args

    self.word_enc = WordEncoder.from_pretrained(args.model_type, return_dict=True)
    for p in self.word_enc.parameters():
      p.requires_grad = False

    self.word_linear = nn.Linear(args.model_dim, args.num_aspects, bias=False)

    self.word_key = nn.Parameter(torch.Tensor(args.model_dim))
    self.word_transform = nn.Linear(args.model_dim, args.model_dim)

    self.sent_key = nn.Parameter(torch.Tensor(args.model_dim))

    self.dropout = nn.Dropout(0.5)

    nn.init.normal_(self.word_key)
    nn.init.normal_(self.sent_key)


  def forward(self, x_BxSxT, y_true_BxC=None, p_true_BxSxTxC=None, step=None):
    B, S, T = x_BxSxT.size()
    H = self.args.num_heads
    D = self.args.model_dim
    E = D // H
    eps = -1e9

    # get word encodings
    x_BSxT = x_BxSxT.view(B*S, T)
    x_mask_BSxT = torch.where(x_BSxT != 0, 1, 0)
    x_BSxTxD = self.word_enc(x_BSxT, x_mask_BSxT, output_hidden_states=True).last_hidden_state
    if self.training:
      assert step is not None
      drop_rate = max(0.2 * (step - self.args.no_warmup_steps) / float(self.args.no_warmup_steps), 0)
      drop_rate = min(drop_rate, 0.2)

      drop_BSxTx1 = torch.rand(B*S, T, 1).cuda()
      drop_BSxTx1 = torch.where(drop_BSxTx1 > drop_rate, 1, 0)
      x_BSxTxD = x_BSxTxD * drop_BSxTx1

    x_BSxTxD = self.dropout(x_BSxTxD)

    # word-level predictions
    p_BSxTxC = torch.tanh(self.word_linear(x_BSxTxD))
    p_BxSxTxC = p_BSxTxC.view(B, S, T, -1)

    # word-level representation/value
    z_BSxTxD = torch.tanh(self.word_transform(x_BSxTxD))
    
    z_list_BSxTxE = z_BSxTxD.chunk(H, -1)
    z_key_list_E = self.word_key.chunk(H, -1)

    q_list_BxSxC = []
    h_list_BxSxE = []
    p_wt_list_BxSxT = []
    for z_BSxTxE, z_key_E in zip(z_list_BSxTxE, z_key_list_E):
      a_BSxT = torch.matmul(z_BSxTxE, z_key_E)
      a_BSxT = a_BSxT.masked_fill(x_mask_BSxT == 0, eps)
      a_BSxT = F.softmax(a_BSxT, -1)
      p_wt_list_BxSxT.append(a_BSxT.view(B, S, T))

      # sentence-level predictions
      q_BSxC = torch.sum(p_BSxTxC * a_BSxT.unsqueeze(-1), 1)
      q_list_BxSxC.append(q_BSxC.view(B, S, -1))

      # sentence-level encodings
      h_BSxE = torch.sum(z_BSxTxE * a_BSxT.unsqueeze(-1), 1)
      h_list_BxSxE.append(h_BSxE.view(B, S, -1))

    q_BxSxHxC = torch.stack(q_list_BxSxC, -2)
    q_BxSxC = q_BxSxHxC.max(dim=-2)[0]

    p_wt_BxSxT = torch.stack(p_wt_list_BxSxT, -2).max(-2)[0]

    h_BxSxHxE = torch.stack(h_list_BxSxE, -2)
    h_BxSxD = h_BxSxHxE.view(B, S, D)

    # sentence-level attention weights
    x_mask_BxS = x_mask_BSxT.view(B, S, T).sum(dim=-1)
    x_mask_BxS = torch.where(x_mask_BxS != 0, 1, 0)
    h_list_BxSxE = h_BxSxD.chunk(H, -1)
    h_key_list_E = self.sent_key.chunk(H, -1)

    y_list_BxC = []
    q_wt_list_BxS = []
    for h_BxSxE, h_key_E in zip(h_list_BxSxE, h_key_list_E):
      b_BxS = torch.matmul(h_BxSxE, h_key_E)
      b_BxS = b_BxS.masked_fill(x_mask_BxS == 0, eps)
      b_BxS = F.softmax(b_BxS, -1)
      q_wt_list_BxS.append(b_BxS)
    
      # document-level predictions
      y_BxC = torch.sum(q_BxSxC * b_BxS.unsqueeze(-1), 1)
      y_list_BxC.append(y_BxC)

    y_BxHxC = torch.stack(y_list_BxC, -2)
    y_BxC = y_BxHxC.max(dim=-2)[0]

    q_wt_BxS = torch.stack(q_wt_list_BxS, -2).max(-2)[0]

    if y_true_BxC is not None:
      eps = 1e-9
      loss_BxC = torch.log(1 + torch.exp(-y_BxC * y_true_BxC))
      loss = loss_BxC.sum(dim=-1).mean()
    else:
      loss = None

    if p_true_BxSxTxC is not None:
      p_true_mask_BxSxTxC = torch.where(p_true_BxSxTxC != 0, 1, 0)
      reg_loss_BxSxTxC = torch.log(1 + torch.exp(-p_BxSxTxC * p_true_BxSxTxC)) * p_true_mask_BxSxTxC
      reg_loss = reg_loss_BxSxTxC.view(B, -1).sum(dim=-1).mean()
    else:
      reg_loss = None

    return {
      'document': y_BxC,
      'sentence': q_BxSxC,
      'word': p_BxSxTxC,
      'loss': loss,
      'reg_loss': reg_loss,
      'sentence_weight': q_wt_BxS,
      'word_weight': p_wt_BxSxT,
    }