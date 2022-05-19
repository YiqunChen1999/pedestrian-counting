_base_ = [
    '../_base_/datasets/dhd_ped_traffic.py',
    '../_base_/schedules/schedule_3x.py', 
    '../_base_/models/atss_r50_fpn_deform_head.py',
    '../_base_/default_runtime.py', 
]
# optimizer
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
