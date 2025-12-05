_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/dior.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]

# model
model = dict(
    backbone=dict(
        depth=18,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')),
    neck=dict(in_channels=[64, 128, 256, 512]), 
    bbox_head=dict(num_classes=20),
    )
# optim_wrapper = dict(
#     optimizer=dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001))