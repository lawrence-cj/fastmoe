"""
#-*- coding:utf-8 -*-
@Time    : 2022/12/2 21:08
@Author  : lawrencechen
@contact : cjs1020440147@icloud.com
@Software: Pycharm Professional Version

"""
from mmdet3d.ops import SparseBasicBlock, make_sparse_convmodule

conv_out = make_sparse_convmodule(
    128,
    128,
    kernel_size=(1, 1, 3),
    stride=(1, 1, 2),
    norm_cfg=dict(type="BN1d", eps=1e-3, momentum=0.01),
    padding=0,
    indice_key="spconv_down2",
    conv_type="SparseConv3d",
)