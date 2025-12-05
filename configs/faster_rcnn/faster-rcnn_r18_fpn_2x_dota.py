_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/dota.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]
model = dict(
    backbone=dict(
        depth=18,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet18')),
    neck=dict(in_channels=[64, 128, 256, 512]),
    roi_head=dict(
        bbox_head=dict(
            num_classes=15,
            )),
    )