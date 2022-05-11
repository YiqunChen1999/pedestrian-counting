_base_ = [
    './atss_r50_fpn_1x_dhd_ped_campus.py'
]
model=dict(
    neck=dict(
        type="FPN_dcd"
    )
)
find_unused_parameters = True