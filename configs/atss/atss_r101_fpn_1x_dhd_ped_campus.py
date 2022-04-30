_base_ = './atss_r50_fpn_1x_dhd_ped_campus.py'
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))
