import torch
import torch.nn as nn
from timm.models.layers import DropPath
import numpy as np
import torch.nn.functional as F
from models.SwinTransformers import SwinTransformer


def conv3x3_bn_relu(in_planes, out_planes, k=3, s=1, p=1, b=False):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=k, stride=s, padding=p, bias=b),
            nn.BatchNorm2d(out_planes),
            nn.GELU(),
            )



class CPNet(nn.Module):
    def __init__(self):
        super(CPNet, self).__init__()

        self.rgb_swin = SwinTransformer(embed_dim=128, depths=[2,2,18,2], num_heads=[4,8,16,32])
        self.depth_swin = SwinTransformer(embed_dim=128, depths=[2,2,18,2], num_heads=[4,8,16,32])
        self.up2 = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor = 4)

        self.CA_SA_Enhance_1 = HamiltonFusion(1024, 1024, 32, spatial_size=(12, 12))
        self.CA_SA_Enhance_2 = HamiltonFusion(512, 512, 16, spatial_size=(24, 24))
        self.CA_SA_Enhance_3 = HamiltonFusion(256, 256, 8, spatial_size=(48, 48))
        self.CA_SA_Enhance_4 = HamiltonFusion(128, 128, 4, spatial_size=(96, 96))

        self.FA_Block2 = Block(dim=256)
        self.FA_Block3 = Block(dim=128)
        self.FA_Block4 = Block(dim=64)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.deconv_layer_1 =  nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.GELU(),
            self.upsample2
        )
        self.deconv_layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
            self.upsample2
        )
        self.deconv_layer_3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            self.upsample2
        )
        self.deconv_layer_4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            self.upsample2
        )
        self.predict_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            self.upsample2,
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1, bias=True),
            )
        self.predtrans2 = nn.Conv2d(128, 1, kernel_size=3, padding=1)
        self.predtrans3 = nn.Conv2d(256, 1, kernel_size=3, padding=1)
        self.predtrans4 = nn.Conv2d(512, 1, kernel_size=3, padding=1)
        self.dwc3 = conv3x3_bn_relu(256, 128)
        self.dwc2 = conv3x3_bn_relu(512, 256)
        self.dwc1 = conv3x3_bn_relu(1024, 512)
        self.dwcon_1 = conv3x3_bn_relu(2048, 1024)
        self.dwcon_2 = conv3x3_bn_relu(1024, 512)
        self.dwcon_3 = conv3x3_bn_relu(512, 256)
        self.dwcon_4 = conv3x3_bn_relu(256, 128)
        self.conv43 = conv3x3_bn_relu(128, 256, s=2)
        self.conv32 = conv3x3_bn_relu(256, 512, s=2)
        self.conv21 = conv3x3_bn_relu(512, 1024, s=2)



    def forward(self,x ,d):
        rgb_list = self.rgb_swin(x)
        depth_list = self.depth_swin(d)

        r4 = rgb_list[0]
        r3 = rgb_list[1]
        r2 = rgb_list[2]
        r1 = rgb_list[3]
        d4 = depth_list[0]
        d3 = depth_list[1]
        d2 = depth_list[2]
        d1 = depth_list[3]

        r3_up = F.interpolate(self.dwc3(r3), size=96, mode='bilinear')
        r2_up = F.interpolate(self.dwc2(r2), size=48, mode='bilinear')
        r1_up = F.interpolate(self.dwc1(r1), size=24, mode='bilinear')
        d3_up = F.interpolate(self.dwc3(d3), size=96, mode='bilinear')
        d2_up = F.interpolate(self.dwc2(d2), size=48, mode='bilinear')
        d1_up = F.interpolate(self.dwc1(d1), size=24, mode='bilinear')

        r1_con = torch.cat((r1, r1), 1)
        r1_con = self.dwcon_1(r1_con)
        d1_con = torch.cat((d1, d1), 1)
        d1_con = self.dwcon_1(d1_con)
        r2_con = torch.cat((r2, r1_up), 1)
        r2_con = self.dwcon_2(r2_con)
        d2_con = torch.cat((d2, d1_up), 1)
        d2_con = self.dwcon_2(d2_con)
        r3_con = torch.cat((r3, r2_up), 1)
        r3_con = self.dwcon_3(r3_con)
        d3_con = torch.cat((d3, d2_up), 1)
        d3_con = self.dwcon_3(d3_con)
        r4_con = torch.cat((r4, r3_up), 1)
        r4_con = self.dwcon_4(r4_con)
        d4_con = torch.cat((d4, d3_up), 1)
        d4_con = self.dwcon_4(d4_con)


        xf_1 = self.CA_SA_Enhance_1(r1_con, d1_con)  
        xf_2 = self.CA_SA_Enhance_2(r2_con, d2_con)  
        xf_3 = self.CA_SA_Enhance_3(r3_con, d3_con)  
        xf_4 = self.CA_SA_Enhance_4(r4_con, d4_con)  


        df_f_1 = self.deconv_layer_1(xf_1)

        xc_1_2 = torch.cat((df_f_1, xf_2), 1)
        df_f_2 = self.deconv_layer_2(xc_1_2)
        df_f_2 = self.FA_Block2(df_f_2)

        xc_1_3 = torch.cat((df_f_2, xf_3), 1)
        df_f_3 = self.deconv_layer_3(xc_1_3)
        df_f_3 = self.FA_Block3(df_f_3)

        xc_1_4 = torch.cat((df_f_3, xf_4), 1)
        df_f_4 = self.deconv_layer_4(xc_1_4)
        df_f_4 = self.FA_Block4(df_f_4)

        y1 = self.predict_layer_1(df_f_4)
        y2 = F.interpolate(self.predtrans2(df_f_3), size=384, mode='bilinear')
        y3 = F.interpolate(self.predtrans3(df_f_2), size=384, mode='bilinear')
        y4 = F.interpolate(self.predtrans4(df_f_1), size=384, mode='bilinear')

        return y1,y2,y3,y4

    def load_pre(self, pre_model):
        self.rgb_swin.load_state_dict(torch.load(pre_model)['model'],strict=False)
        print(f"RGB SwinTransformer loading pre_model ${pre_model}")
        self.depth_swin.load_state_dict(torch.load(pre_model)['model'], strict=False)
        print(f"Depth SwinTransformer loading pre_model ${pre_model}")


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SA_Enhance(nn.Module):
    def __init__(self, kernel_size=7):
        super(SA_Enhance, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)


class FFTAttention2D(nn.Module):
    def __init__(self, num_heads=8, spatial_size=None, window='hann', use_learnable_filter=True, fft_norm="backward", eps=1e-6):  # ab: fft_norm='backward', "ortho"
        super().__init__()
        self.num_heads = num_heads
        self.window = window
        self.use_filter = use_learnable_filter
        self.fft_norm = fft_norm
        self.eps = eps

        self.spatial_size = tuple(spatial_size)
        h, w = self.spatial_size
        if self.use_filter:
            self.freq_response = nn.Parameter(torch.ones(1, 1, 1, h, w // 2 + 1))
        else:
            self.register_parameter('freq_response', None)

    def _make_window(self, h, w, device, dtype):
        if self.window is None:
            return torch.ones((h, w), device=device, dtype=dtype)
        if self.window != 'hann':
            raise ValueError(f"Unsupported window type: {self.window}")
        if h == 1:
            win_h = torch.ones(1, device=device, dtype=dtype)
        else:
            win_h = torch.hann_window(h, periodic=False, device=device, dtype=dtype)
        if w == 1:
            win_w = torch.ones(1, device=device, dtype=dtype)
        else:
            win_w = torch.hann_window(w, periodic=False, device=device, dtype=dtype)
        return win_h.view(h, 1) * win_w.view(1, w)

    def forward(self, q, k, h, w):
        B, Hh, N, C = q.shape
        device, dtype = q.device, q.dtype

        # FFT in fp32 is safer under AMP / fp16 / bf16 training.
        if q.dtype in (torch.float16, torch.bfloat16):
            fft_dtype = torch.float32
        else:
            fft_dtype = q.dtype

        q_sp = q.permute(0, 1, 3, 2).contiguous().reshape(B, Hh, C, h, w).to(dtype=fft_dtype)
        k_sp = k.permute(0, 1, 3, 2).contiguous().reshape(B, Hh, C, h, w).to(dtype=fft_dtype)
        win = self._make_window(h, w, device, dtype).view(1, 1, 1, h, w)

        q_freq = torch.fft.rfft2(q_sp * win, s=(h, w), dim=(-2, -1), norm=self.fft_norm)
        k_freq = torch.fft.rfft2(k_sp * win, s=(h, w), dim=(-2, -1), norm=self.fft_norm)
        cross = q_freq * k_freq.conj()

        if self.use_filter:
            cross = cross * self.freq_response.to(device=device, dtype=cross.real.dtype)

        corr = torch.fft.irfft2(cross, s=(h, w), dim=(-2, -1), norm=self.fft_norm)

        # ab
        # energy = (win ** 2).sum(dim=(-2, -1), keepdim=True).clamp_min(self.eps)
        # corr = corr / energy.sqrt()

        corr = (
            corr.reshape(B, Hh, C, N)
            .permute(0, 1, 3, 2)
            .contiguous()
            .to(dtype=dtype)
        )

        return corr


class HamiltonFusion(nn.Module):
    def __init__(self, dim=64, out_dim=64, heads=8, T=1, gamma=0.4, spatial_size=None):
        super().__init__()

        self.heads = heads
        self.head_dim = dim // heads
        self.T = T
        self.gamma = gamma

        self.to_qkv_rgb   = nn.Linear(dim, dim * 3, bias=False)
        self.to_qkv_depth = nn.Linear(dim, dim * 3, bias=False)

        self.fft_attn = FFTAttention2D(heads, spatial_size=spatial_size, window='hann', use_learnable_filter=True)

        self.alpha_aux = nn.Parameter(torch.tensor(0.8))   
        self.alpha_main = nn.Parameter(torch.tensor(0.2))  

        self.gate_mlp = nn.Sequential(
            nn.Linear(self.head_dim * 2, self.head_dim), nn.GELU(),
            nn.Linear(self.head_dim, self.head_dim), nn.Sigmoid()
        )

        self.proj_out = nn.Linear(dim, out_dim)
        self.proj_out_cat = nn.Linear(dim * 2, out_dim)

        self.scale = self.head_dim ** -0.5

    def forward(self, rgb, depth):
        """
        rgb:   [B, C, H, W]  
        depth: [B, C, H, W]  
        """
        B, C, H, W = rgb.shape
        N = H * W
        device = rgb.device

        x_rgb   = rgb.permute(0,2,3,1).reshape(B, N, C)
        x_depth = depth.permute(0,2,3,1).reshape(B, N, C)

        qkv_r = self.to_qkv_rgb(x_rgb).reshape(B, N, 3, self.heads, -1).permute(2,0,3,1,4)
        qr, kr, vr = qkv_r[0], qkv_r[1], qkv_r[2]
        qkv_d = self.to_qkv_depth(x_depth).reshape(B, N, 3, self.heads, -1).permute(2,0,3,1,4)
        qd, kd, vd = qkv_d[0], qkv_d[1], qkv_d[2]

        q = qr   # [B, H, N, c]
        p = torch.zeros_like(q)

        F_main = self.fft_attn(q, kd, H, W)  
        F_aux  = self.fft_attn(q, kr, H, W)       

        for _ in range(self.T):
            force = (F_main - q) + self.gamma * (q - F_aux)

            bHn, r = (q.shape[0]*q.shape[1]*q.shape[2], q.shape[3])
            qg = q.reshape(bHn, r)   
            pg = p.reshape(bHn, r)
            gate_in = torch.cat([qg, pg], dim=-1)
            g = self.gate_mlp(gate_in).reshape(B, self.heads, N, r) 

            # p = p + g * force
            p = (1-g)*p + g * force
            q = q + p

        q_up = q.transpose(1,2).reshape(B, N, C)

        out_rgb   = x_rgb   + self.alpha_aux  * q_up
        out_depth = x_depth + self.alpha_main * q_up

        fused = torch.cat([out_rgb, out_depth], dim=2)  
        out = self.proj_out_cat(fused)
        out = out.reshape(B, H, W, -1).permute(0,3,1,2)
        return out



def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class Block(nn.Module):
    def __init__(self, dim, drop_rate=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim,)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C]
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]

        x = shortcut + self.drop_path(x)
        return x
