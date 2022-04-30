_base_ = [
    './atss_r50_fpn_1x_coco.py'
]
model=dict(
    neck=dict(
        type="FPN_dcd"
    )
)
find_unused_parameters = True