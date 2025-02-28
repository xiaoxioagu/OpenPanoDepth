import copy
import functools
import time

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from .equi_pers.equi2pers import equi2pers
from .equi_pers.pers2equi import pers2equi
from torch.nn.modules import padding
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
import cv2
from .ViT import miniViT, layers
from .ViT.layers import PixelWiseDotProduct
from .blocks import Transformer_Block
import matplotlib.pyplot as plt

color = {"0": [255, 218, 185], "1": [187, 255, 255], "2": [0, 0, 0],
         "3": [25, 25, 112], "4": [127, 255, 212], "5": [173, 255, 47],
         "6": [255, 255, 0], "7": [119, 136, 153], "8": [205, 92, 92],
         "9": [255, 165, 0], "10": [255, 105, 180], "11": [138, 43, 226],
         "12": [255, 250, 205], "13": [205, 0, 0], "14": [0, 0, 255],
         "15": [122, 55, 139], "16": [0, 178, 238], "17": [238, 121, 159]
         }


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ConvBnReLU_v2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU_v2, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, (kernel_size, kernel_size, 1),
                              stride=(stride, stride, 1), padding=(pad, pad, 0), bias=False, padding_mode='zeros')
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ConvBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBn, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = ConvBnReLU(in_channels, out_channels, kernel_size=3, stride=stride, pad=1)
        self.conv2 = ConvBn(out_channels, out_channels, kernel_size=3, stride=1, pad=1)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        return out


class SharedMLP(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            transpose=False,
            padding_mode='zeros',
            bn=False,
            activation_fn=None
    ):
        super(SharedMLP, self).__init__()

        conv_fn = nn.ConvTranspose2d if transpose else nn.Conv2d

        self.conv = conv_fn(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding_mode=padding_mode
        )
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-6, momentum=0.99) if bn else None
        self.activation_fn = activation_fn

    def forward(self, input):
        r"""
            Forward pass of the network
            Parameters
            ----------
            input: torch.Tensor, shape (B, d_in, N, K)
            Returns
            -------
            torch.Tensor, shape (B, d_out, N, K)
        """
        x = self.conv(input)
        if self.batch_norm:
            x = self.batch_norm(x)
        if self.activation_fn:
            x = self.activation_fn(x)
        return x


def convert_conv(layer):
    for name, module in layer.named_modules():
        if name:
            try:
                sub_layer = getattr(layer, name)
                if isinstance(sub_layer, nn.Conv2d):
                    m = copy.deepcopy(sub_layer)
                    new_layer = nn.Conv3d(m.in_channels, m.out_channels,
                                          kernel_size=(m.kernel_size[0], m.kernel_size[1], 1),
                                          stride=(m.stride[0], m.stride[1], 1), padding=(m.padding[0], m.padding[1], 0),
                                          padding_mode='zeros', bias=False)
                    new_layer.weight.data.copy_(m.weight.data.unsqueeze(-1))
                    if m.bias is not None:
                        new_layer.bias.data.copy_(m.bias.data)
                    layer._modules[name] = copy.deepcopy(new_layer)
            except AttributeError:
                name = name.split('.')[0]
                sub_layer = getattr(layer, name)
                sub_layer = convert_conv(sub_layer)
                layer.__setattr__(name=name, value=sub_layer)
    return layer


def convert_bn(layer):
    for name, module in layer.named_modules():
        if name:
            try:
                sub_layer = getattr(layer, name)
                if isinstance(sub_layer, nn.BatchNorm2d):
                    m = copy.deepcopy(sub_layer)
                    new_layer = nn.BatchNorm3d(m.num_features)
                    new_layer.weight.data.copy_(m.weight.data)
                    new_layer.bias.data.copy_(m.bias.data)
                    new_layer.running_mean.data.copy_(m.running_mean.data)
                    new_layer.running_var.data.copy_(m.running_var.data)
                    layer._modules[name] = copy.deepcopy(new_layer)
            except AttributeError:
                name = name.split('.')[0]
                sub_layer = getattr(layer, name)
                sub_layer = convert_bn(sub_layer)
                layer.__setattr__(name=name, value=sub_layer)
    return layer


class Transformer_cascade(nn.Module):
    def __init__(self, emb_dims, num_patch, depth, num_heads):
        super(Transformer_cascade, self).__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(emb_dims, eps=1e-6)
        self.pos_emb = nn.Parameter(torch.zeros(1, num_patch, emb_dims))
        nn.init.trunc_normal_(self.pos_emb, std=.02)
        for _ in range(depth):
            layer = Transformer_Block(emb_dims, num_heads=num_heads)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, x):
        hidden_states = x + self.pos_emb
        for i, layer_block in enumerate(self.layer):
            hidden_states = layer_block(hidden_states)

        encoded = self.encoder_norm(hidden_states)
        return encoded


class Global_network(nn.Module):
    def __init__(self):
        super(Global_network, self).__init__()
        pretrain_model = torchvision.models.resnet34(pretrained=True)

        encoder = pretrain_model

        self.conv1 = encoder.conv1
        self.bn1 = encoder.bn1
        self.relu = nn.ReLU(True)
        self.layer1 = encoder.layer1  # 64
        self.layer2 = encoder.layer2  # 128
        self.layer3 = encoder.layer3  # 256
        self.layer4 = encoder.layer4  # 512

        self.down = nn.Conv2d(512, 512 // 4, kernel_size=1, stride=1, padding=0)

        self.de_conv0_0 = ConvBnReLU(512 + 128, 256, kernel_size=3, stride=1)
        self.de_conv0_1 = ConvBnReLU(256 + 256, 128, kernel_size=3, stride=1)
        self.de_conv1_0 = ConvBnReLU(128, 128, kernel_size=3, stride=1)
        self.de_conv1_1 = ConvBnReLU(128 + 128, 64, kernel_size=3, stride=1)
        self.de_conv2_0 = ConvBnReLU(64, 64, kernel_size=3, stride=1)
        self.de_conv2_1 = ConvBnReLU(64 + 64, 64, kernel_size=3, stride=1)
        self.de_conv3_0 = ConvBnReLU(64, 64, kernel_size=3, stride=1)
        self.de_conv3_1 = ConvBnReLU(64 + 64, 32, kernel_size=3, stride=1)
        self.de_conv4_0 = ConvBnReLU(32, 32, kernel_size=3, stride=1)

        self.transformer = Transformer_cascade(128, 8 * 16, depth=6, num_heads=4)

    def forward(self, rgb):
        bs, c, erp_h, erp_w = rgb.shape
        conv1 = self.relu(self.bn1(self.conv1(rgb)))
        pool = F.max_pool2d(conv1, kernel_size=3, stride=2, padding=1)

        layer1 = self.layer1(pool)

        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        layer4_reshape = self.down(layer4)

        layer4_reshape = layer4_reshape.permute(0, 2, 3, 1).reshape(bs, 8 * 16, -1)
        layer4_reshape = self.transformer(layer4_reshape)

        layer4_reshape = layer4_reshape.permute(0, 2, 1).reshape(bs, -1, 8, 16)
        layer4 = torch.cat([layer4, layer4_reshape], 1)

        up = F.interpolate(layer4, size=(layer3.shape[-2], layer3.shape[-1]), mode='bilinear', align_corners=False)

        de_conv0_0 = self.de_conv0_0(up)
        concat = torch.cat([de_conv0_0, layer3], 1)
        de_conv0_1 = self.de_conv0_1(concat)

        up = F.interpolate(de_conv0_1, size=(layer2.shape[-2], layer2.shape[-1]), mode='bilinear', align_corners=False)
        de_conv1_0 = self.de_conv1_0(up)
        concat = torch.cat([de_conv1_0, layer2], 1)
        de_conv1_1 = self.de_conv1_1(concat)

        up = F.interpolate(de_conv1_1, size=(layer1.shape[-2], layer1.shape[-1]), mode='bilinear', align_corners=False)
        de_conv2_0 = self.de_conv2_0(up)
        concat = torch.cat([de_conv2_0, layer1], 1)
        de_conv2_1 = self.de_conv2_1(concat)

        up = F.interpolate(de_conv2_1, size=(conv1.shape[-2], conv1.shape[-1]), mode='bilinear', align_corners=False)
        de_conv3_0 = self.de_conv3_0(up)
        concat = torch.cat([de_conv3_0, conv1], 1)
        de_conv3_1 = self.de_conv3_1(concat)

        # up = F.interpolate(de_conv3_1, (erp_h // 4, erp_w // 4), mode='bilinear')
        de_conv4_0 = self.de_conv4_0(de_conv3_1)

        return de_conv4_0


class Local_network(nn.Module):
    def __init__(self,patchsize):
        super(Local_network, self).__init__()
        pretrain_model = torchvision.models.resnet34(pretrained=True)

        encoder = convert_conv(pretrain_model)
        encoder = convert_bn(encoder)

        self.conv1 = encoder.conv1
        self.bn1 = encoder.bn1

        self.relu = nn.ReLU(True)
        self.layer1 = encoder.layer1  # 64
        self.layer2 = encoder.layer2  # 128
        self.layer3 = encoder.layer3  # 256
        self.layer4 = encoder.layer4  # 512

        self.de_conv0 = ConvBnReLU_v2(128, 64, kernel_size=3, stride=1)
        self.de_conv1 = ConvBnReLU_v2(64, 64, kernel_size=3, stride=1)
        self.de_conv2 = ConvBnReLU_v2(64, 64, kernel_size=3, stride=1)
        self.de_conv3 = ConvBnReLU_v2(64, 128, kernel_size=3, stride=1)

        if patchsize == 256:
            self.down = nn.Conv3d(512, 512 // 4, kernel_size=(8, 8, 1), stride=(8, 8, 1), padding=0) # for TP:256x256
        elif patchsize == 128:
            self.down = nn.Conv3d(512, 512 // 4, kernel_size=(4, 4, 1), stride=(4, 4, 1), padding=0) # for TP:128x128

        self.tangent_layer = miniViT.tangent_ViT(128, n_query_channels=128, patch_size=4,
                                                 dim_out=100,
                                                 embedding_dim=128, norm='linear')

        self.mlp_points = nn.Sequential(
            nn.Conv2d(5, 16, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

    def forward(self, tangent, center_point, uv):
        bs, c, patch_h, patch_w, n_patch = tangent.shape

        device = tangent.device
        rho = torch.ones((uv.shape[0], 1, patch_h // 4, patch_w // 4), dtype=torch.float32, device=device)

        center_points = center_point.to(device)
        center_points = center_points.reshape(-1, 2, 1, 1).repeat(1, 1, patch_h // 4, patch_w // 4)

        new_xyz = torch.cat([center_points, rho, center_points], 1)
        point_feat = self.mlp_points(new_xyz.contiguous())
        point_feat = point_feat.permute(1, 2, 3, 0).unsqueeze(0)

        conv1 = self.relu(self.bn1(self.conv1(tangent)))
        pool = F.max_pool3d(conv1, kernel_size=(3, 3, 1), stride=(2, 2, 1), padding=(1, 1, 0))

        layer1 = self.layer1(pool)

        layer1 = layer1 + point_feat
        layer2 = self.layer2(layer1)

        layer3 = self.layer3(layer2)

        layer4 = self.layer4(layer3)

        layer4_reshape = self.down(layer4)

        tangent_feature_set = layer4_reshape
        layer4_reshape = layer4_reshape.permute(0, 2, 3, 4, 1).flatten(-2).permute(0, 3, 1, 2)
        up = F.interpolate(layer4_reshape, size=(layer4.shape[-3], layer4.shape[-2]), mode='bilinear',
                           align_corners=False)
        up = up.permute(0, 2, 3, 1).reshape(bs, layer4.shape[-3], layer4.shape[-2], -1, n_patch).permute(0, 3, 1, 2, 4)

        deconv_layer4 = self.de_conv0(up)
        deconv_layer4 = deconv_layer4.permute(0, 2, 3, 4, 1).flatten(-2).permute(0, 3, 1, 2)
        up = F.interpolate(deconv_layer4, size=(layer3.shape[-3], layer3.shape[-2]), mode='bilinear',
                           align_corners=False)
        up = up.permute(0, 2, 3, 1).reshape(bs, layer3.shape[-3], layer3.shape[-2], -1, n_patch).permute(0, 3, 1, 2, 4)

        deconv_layer3 = self.de_conv1(up)
        deconv_layer3 = deconv_layer3.permute(0, 2, 3, 4, 1).flatten(-2).permute(0, 3, 1, 2)
        up = F.interpolate(deconv_layer3, size=(layer2.shape[-3], layer2.shape[-2]), mode='bilinear',
                           align_corners=False)
        up = up.permute(0, 2, 3, 1).reshape(bs, layer2.shape[-3], layer2.shape[-2], -1, n_patch).permute(0, 3, 1, 2, 4)

        deconv_layer2 = self.de_conv2(up)
        deconv_layer2 = deconv_layer2.permute(0, 2, 3, 4, 1).flatten(-2).permute(0, 3, 1, 2)
        up = F.interpolate(deconv_layer2, size=(layer1.shape[-3], layer1.shape[-2]), mode='bilinear',
                           align_corners=False)
        up = up.permute(0, 2, 3, 1).reshape(bs, layer1.shape[-3], layer1.shape[-2], -1, n_patch).permute(0, 3, 1, 2, 4)
        deconv_layer1 = self.de_conv3(up)

        bs, c, h, w, n_patch = up.shape

        tangent_embedding_set = []
        tangent_bin_set = []

        for i in range(n_patch):
            tangent_feature = deconv_layer1[..., i]
            tangent_bin, tangent_embedding = self.tangent_layer(tangent_feature)
            tangent_embedding_set.append(tangent_embedding)
            tangent_bin_set.append(tangent_bin)
        tangent_embedding_set = torch.stack(tangent_embedding_set, dim=-1)
        tangent_bin_set = torch.stack(tangent_bin_set, dim=-1)
        return tangent_bin_set, tangent_embedding_set, tangent_feature_set


class hrdfuse(nn.Module):
    def __init__(self, model_cfg):
        nbins = model_cfg['nbins']
        min_val = model_cfg['min_val']
        max_val = model_cfg['max_val']
        nrows = model_cfg['nrows']
        npatches = model_cfg['npatches_dict'][nrows]
        patch_size = model_cfg['patch_size']
        fov = model_cfg['fov']

        self.num_classes = nbins
        self.min_val = min_val
        self.max_val = max_val
        self.nrows = nrows
        self.npatches = npatches
        self.patch_size = patch_size
        self.fov = fov
        super(hrdfuse, self).__init__()

        self.global_network = Global_network()
        self.local_network = Local_network(self.patch_size)

        self.pred = nn.Conv3d(32, 1, (3, 3, 1), 1, padding=(1, 1, 0), padding_mode='zeros')
        self.weight_pred = nn.Conv3d(32, 1, (3, 3, 1), 1, padding=(1, 1, 0), padding_mode='zeros')
        self.min_depth = 0.1
        self.max_depth = 10.0

        self.conv_out_erp = nn.Sequential(nn.Conv2d(128, self.num_classes, kernel_size=1, stride=1, padding=0),
                                          nn.Softmax(dim=1))

        self.conv_out_tangent = nn.Sequential(nn.Conv2d(128, self.num_classes, kernel_size=1, stride=1, padding=0),
                                              nn.Softmax(dim=1))

        self.conv_out = nn.Sequential(nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
                                      nn.Softmax(dim=1))

        self.adaptive_bins_layer = miniViT.mViT(32, n_query_channels=128, patch_size=8,
                                                dim_out=100,
                                                embedding_dim=128, norm='linear')

        self.wisedotlayer = PixelWiseDotProduct()

        self.offset = nn.Sequential(nn.Linear(128, 256),
                                    nn.LeakyReLU(),
                                    nn.Linear(256, 256),
                                    nn.LeakyReLU(),
                                    nn.Linear(256, nbins),
                                    nn.Softmax(dim=1))
        self.w = nn.Parameter(torch.ones(2))

        self.w1 = nn.Parameter(torch.from_numpy(np.array([0.1,0.9])))
    def cal_sim(self, tangent_feature_set, feature_erp, erp_h, erp_w):
        feature_erp = F.interpolate(feature_erp, (erp_h, erp_w), mode='bilinear')

        bs, c, _, _, N = tangent_feature_set.shape
        tangent_feature = tangent_feature_set.reshape(bs, c, N).permute(0, 2, 1)
        tangent_feature_normed = torch.norm(tangent_feature, p=2, dim=-1).unsqueeze(-1).unsqueeze(-1)

        erp_feature_normed = torch.norm(feature_erp, p=2, dim=1).unsqueeze(1)
        similarity_map = torch.einsum('bne, behw -> bnhw', tangent_feature, feature_erp)
        similarity_map = similarity_map / tangent_feature_normed
        similarity_map = similarity_map / erp_feature_normed
        similarity_max_map, similarity_index_map = torch.max(similarity_map, dim=1, keepdim=True)
        one_hot = torch.FloatTensor(similarity_index_map.shape[0], tangent_feature.shape[1],
                                    similarity_index_map.shape[2], similarity_index_map.shape[3]).zero_().to(
            tangent_feature.device)
        similarity_index_map = one_hot.scatter_(1, similarity_index_map, 1)

        return similarity_index_map

    def forward(self, rgb, confidence=True):
        bs, _, erp_h, erp_w = rgb.shape
        device = rgb.device
        patch_h, patch_w = pair(self.patch_size)

        high_res_patch, _, _, _ = equi2pers(rgb, self.fov, self.nrows, patch_size=self.patch_size)
        _, xyz, uv, center_points = equi2pers(rgb, self.fov, self.nrows, patch_size=(patch_h, patch_w))

        center_points = center_points.to(device)

        n_patch = high_res_patch.shape[-1]

        erp_feature = self.global_network(rgb)

        choose_tangent = random.sample(range(n_patch), n_patch)

        tangent_bin_set, tangent_embedding_set, tangent_feature_set = self.local_network(
            high_res_patch[:, :, :, :, choose_tangent],
            center_points[choose_tangent, :], uv)
        tangent_embedding_set = torch.mean(tangent_embedding_set, dim=1, keepdim=False)

        _, bin_widths_normed_erp, range_attention_maps_erp, queries_erp, feature_map_erp = self.adaptive_bins_layer(
            erp_feature)
        range_attention_maps_erp = F.interpolate(range_attention_maps_erp, (erp_h, erp_w), mode='bilinear')

        similarity_map = self.cal_sim(tangent_feature_set, feature_map_erp, erp_h=erp_h, erp_w=erp_w)


        bin_tangent_map = torch.einsum('ben, bnhw -> behw', tangent_bin_set, similarity_map)

        queries_tangent_map = torch.einsum('ben, bnhw -> behw', tangent_embedding_set, similarity_map)
        range_attention_map_tangent = torch.einsum('bse, behw -> bshw', queries_erp, queries_tangent_map)

        out_erp = self.conv_out_erp(range_attention_maps_erp)

        out_tangent = self.conv_out_tangent(self.w1[1]*range_attention_maps_erp+ self.w1[0]* range_attention_map_tangent)
        bin_widths_erp = (self.max_val - self.min_val) * bin_widths_normed_erp  # .shape = N, dim_out
        bin_widths_erp = nn.functional.pad(bin_widths_erp, (1, 0), mode='constant', value=self.min_val)

        bin_widths_tangent = (self.max_val - self.min_val) * bin_tangent_map  # .shape = N, dim_out, h ,w

        bin_widths_tangent = nn.functional.pad(bin_widths_tangent, (0, 0, 0, 0, 1, 0), mode='constant',
                                               value=self.min_val)

        bin_edges_erp = torch.cumsum(bin_widths_erp, dim=1)

        bin_edges_tangent = torch.cumsum(bin_widths_tangent, dim=1)

        centers_erp = 0.5 * (bin_edges_erp[:, :-1] + bin_edges_erp[:, 1:])
        n_erp, dout_erp = centers_erp.size()
        centers_erp = centers_erp.view(n_erp, dout_erp, 1, 1)

        centers_tangent = 0.5 * (bin_edges_tangent[:, :-1] + bin_edges_tangent[:, 1:])


        pred_local = torch.sum(out_tangent * centers_tangent, dim=1, keepdim=True)
        pred_global = torch.sum(out_erp * centers_erp, dim=1, keepdim=True)

        w0 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        w1 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        pred = w0 * pred_global + w1 * pred_local

        return pred


if __name__ == "__main__":
    net = spherical_fusion(4, 18)
    input = torch.randn((3, 3, 512, 1024), dtype=torch.float32)
    output, bin_edges = net(input)
    print(output.shape)
