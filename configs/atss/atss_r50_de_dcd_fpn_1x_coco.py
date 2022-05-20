_base_ = [
    './atss_r50_fpn_1x_dhd_ped_traffic.py'
]
model=dict(
    neck=dict(
        type='de_FPN_dcd',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5,
        # K代表每层中，动态卷积的次数
        # t代表softmax函数中的温度
        K=4,
        t=30
    )
)
find_unused_parameters = True