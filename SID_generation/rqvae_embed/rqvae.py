# Copyright (c) 2022-present, Kakao Brain Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import nn
from torch.nn import functional as F

from .layers import ResnetBlock
from .modules import Encoder, Decoder
from .quantizations import RQBottleneck


class RQVAE_EMBED(nn.Module):
    def __init__(self,
                 *,
                 embed_dim=64,
                 n_embed=512,
                 decay=0.99,
                 loss_type='mse',
                 latent_loss_weight=0.25,
                 bottleneck_type='rq',
                 ddconfig=None,
                 checkpointing=False,
                 VQ_ema=True,
                 latent_weight=[1, 0.5],
                 do_bn=False,
                 **kwargs):
        super().__init__()

        assert loss_type in ['mse', 'l1', 'cosine']

        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.bn = nn.BatchNorm1d(embed_dim)
        self.do_bn = do_bn

        def set_checkpointing(m):
            if isinstance(m, ResnetBlock):
                m.checkpointing = checkpointing

        # self.encoder.apply(set_checkpointing)
        # self.decoder.apply(set_checkpointing)

        if bottleneck_type == 'rq':
            latent_shape = kwargs['latent_shape']
            code_shape = kwargs['code_shape']
            shared_codebook = kwargs['shared_codebook']
            restart_unused_codes = kwargs['restart_unused_codes']
            # self.quantizer = HierRQBottleneck(latent_shape=latent_shape,
            self.quantizer = RQBottleneck(latent_shape=latent_shape,
                                          code_shape=code_shape,
                                          n_embed=n_embed,
                                          decay=decay,
                                          shared_codebook=shared_codebook,
                                          restart_unused_codes=restart_unused_codes,
                                          commitment_loss=loss_type,
                                          VQ_ema=VQ_ema,
                                          latent_weight=latent_weight,
                                          rotation_trick=kwargs.get('rotation_trick', False),
                                          )
            self.code_shape = code_shape
        else:
            raise ValueError("invalid 'bottleneck_type' (must be 'rq')")

        # self.quant_conv = nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        # self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

        self.loss_type = loss_type
        self.latent_loss_weight = latent_loss_weight

    def forward(self, xs, num_samples=0, detail=False, reference_code=None, **kwargs):
        # 1. 编码输入特征 xs，得到潜在表示 z_e
        z_e = self.encode(xs)

        # 2. 使用量化器对 z_e 进行残差量化
        #    - z_q: 量化后的特征
        #    - quant_loss: 量化损失（包含承诺损失等）
        #    - code: 量化后的码本索引
        #    - feature_norm: 输入特征的范数
        #    - quant_norm: 量化特征的范数
        #    - angle: 量化特征与残差的夹角
        #    - angle_aggregated: 累计量化特征与原始特征的夹角
        #    - z_q_neg: 负样本量化特征（如果 num_samples > 0）
        #    - all_distances: 所有码本的距离
        z_q, quant_loss, code, feature_norm, quant_norm, angle, angle_aggregated, z_q_neg, all_distances = self.quantizer(
            z_e, num_samples=num_samples, reference_code=reference_code, **kwargs)
        
        # 3. 如果不需要详细损失信息，则只保留承诺损失
        if not detail:
            quant_loss = quant_loss['commitment_loss']

        # 4. 解码量化特征 z_q，得到重构特征 out
        #    如果需要生成负样本（num_samples > 0），则同时解码正样本和负样本
        bs, level = xs.shape
        if num_samples > 0:
            # 4a. 将正样本和负样本的量化特征拼接，一起解码
            z_combined = torch.cat((z_q, z_q_neg), dim=0)
            decoded = self.decode(z_combined)  # 解码
            out = decoded[:bs]                 # 正样本重构特征
            out_neg = decoded[bs:]             # 负样本重构特征
            out_neg = out_neg.reshape(bs, num_samples, -1) # 调整负样本形状
            
            # 4b. 返回正样本重构特征、量化损失、码本索引等信息
            return out, quant_loss, code, feature_norm, quant_norm, z_q, out_neg
        else:
            # 4c. 直接解码量化特征 z_q
            out = self.decode(z_q)  # 解码
            
            # 4d. 返回重构特征、量化损失、码本索引等信息
            return out, quant_loss, code, feature_norm, quant_norm, z_q, all_distances, z_e

    def encode(self, x):
        z_e = self.encoder(x)
        # z_e = F.normalize(z_e, p=2, dim=1)
        z_e = z_e.contiguous()
        return z_e

    def decode(self, z_q):
        z_q = z_q.contiguous()
        out = self.decoder(z_q)
        # out = F.normalize(out, p=2, dim=1)
        return out

    @torch.no_grad()
    def get_codes(self, xs):
        z_e = self.encode(xs)
        # if self.do_bn:
        #     z_e = self.bn(z_e)
        _, _, code, _, _, _, _, _, _ = self.quantizer(z_e)
        return code

    @torch.no_grad()
    def get_soft_codes(self, xs, temp=1.0, stochastic=False):
        assert hasattr(self.quantizer, 'get_soft_codes')

        z_e = self.encode(xs)
        soft_code, code = self.quantizer.get_soft_codes(z_e, temp=temp, stochastic=stochastic)
        return soft_code, code

    @torch.no_grad()
    def decode_code(self, code):
        z_q = self.quantizer.embed_code(code)
        decoded = self.decode(z_q)
        return decoded

    def get_recon_imgs(self, xs_real, xs_recon):

        xs_real = xs_real * 0.5 + 0.5
        xs_recon = xs_recon * 0.5 + 0.5
        xs_recon = torch.clamp(xs_recon, 0, 1)

        return xs_real, xs_recon

    def cosine_loss(self, x1, x2):
        cos_sim = F.cosine_similarity(x1, x2, dim=1)
        loss = 1 - cos_sim
        return loss.mean()

    def compute_loss(self, out, quant_loss, code, xs=None, valid=False, ):

        # 根据配置的损失类型计算重构损失
        # 支持三种损失类型：MSE、L1、余弦相似度
        if self.loss_type == 'mse':
            # 均方误差损失: L_recon = E[(out - xs)^2]
            loss_recon = F.mse_loss(out, xs, reduction='mean')
        elif self.loss_type == 'l1':
            # L1损失: L_recon = E[|out - xs|]
            loss_recon = F.l1_loss(out, xs, reduction='mean')
        elif self.loss_type == 'cosine':
            # 余弦相似度损失: L_recon = 1 - cos(out, xs)
            loss_recon = self.cosine_loss(out, xs)
        else:
            raise ValueError('incompatible loss type')

        # 潜在空间损失即为量化器返回的损失
        # 对于RQ-VAE，这通常是承诺损失(commitment loss)
        loss_latent = quant_loss

        # 在验证模式下，对损失进行缩放以正确计算批次平均值
        if valid:
            loss_recon = loss_recon * xs.shape[0]
            loss_latent = loss_latent * xs.shape[0]

        # 总损失 = 重构损失 + 潜在空间损失
        # L_total = L_recon + β * L_latent
        # 其中β是潜在损失权重，默认为0.25
        loss_total = loss_recon + loss_latent

        return {
            'loss_total': loss_total,
            'recon_loss': loss_recon,
            'loss_latent': loss_latent,
            # 'codes': [code]
        }

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def get_code_emb_with_depth(self, code):
        return self.quantizer.embed_code_with_depth(code)

    @torch.no_grad()
    def decode_partial_code(self, code, code_idx, decode_type='select'):
        r"""
        Use partial codebooks and decode the codebook features.
        If decode_type == 'select', the (code_idx)-th codebook features are decoded.
        If decode_type == 'add', the [0,1,...,code_idx]-th codebook features are added and decoded.
        """
        z_q = self.quantizer.embed_partial_code(code, code_idx, decode_type)
        decoded = self.decode(z_q)
        return decoded

    @torch.no_grad()
    def forward_partial_code(self, xs, code_idx, decode_type='select'):
        r"""
        Reconstuct an input using partial codebooks.
        """
        code = self.get_codes(xs)
        out = self.decode_partial_code(code, code_idx, decode_type)
        return out


if __name__ == '__main__':
    model = RQVAE_EMBED(n_embed=48, latent_shape=[8, 8, 64], code_shape=[8, 8, 3])
    model.to('cuda')
    # model
    input = torch.rand(4096, 256)
    model(input.to('cuda'))
