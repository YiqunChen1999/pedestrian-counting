_base_ = [
    '../_base_/models/atss_r50_fpn.py',
    '../common/mstrain_3x_dhd_ped_traffic.py', 
]
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)