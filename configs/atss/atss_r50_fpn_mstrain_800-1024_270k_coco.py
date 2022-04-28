_base_ = [
    '../_base_/models/atss_r50_fpn.py',
    '../common/mstrain_270k_coco.py', 
]
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)