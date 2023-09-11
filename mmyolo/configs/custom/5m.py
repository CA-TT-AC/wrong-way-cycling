_base_ = '../yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py'

data_root = 'data/Ver3/output/'  # Root path of data
# Path of train annotation file
train_ann_file = 'train.json'
train_data_prefix = 'train/'  # Prefix of train image path
# Path of val annotation file
val_ann_file = 'val.json'
val_data_prefix = 'val/'  # Prefix of val image path

class_name = ('ebike', ) # the name of classes
num_classes = len(class_name) # the number of classes
metainfo = dict(classes=class_name, palette=[(20, 220, 60)])

train_batch_size_per_gpu = 32
max_epochs = 400  # Maximum training epochs

# ========================modified parameters======================
deepen_factor = 0.67
widen_factor = 0.75
lr_factor = 0.1
affine_scale = 0.9
loss_cls_weight = 0.3
loss_obj_weight = 0.7
mixup_prob = 0.1
load_from = "/mnt/c/Users/ShengRen/wise_transportation/mmyolo/work_dirs/MMYolo/yolov5_m-v61_syncbn_fast_8xb16-300e_coco_20220917_204944-516a710f.pth"

# =======================Unmodified in most cases==================
num_det_layers = _base_.num_det_layers
img_scale = _base_.img_scale

model = dict(
    backbone=dict(
        frozen_stages=4,
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
    ),
    neck=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
    ),
    bbox_head=dict(
        head_module=dict(widen_factor=widen_factor,
                         num_classes=num_classes),
        loss_cls=dict(loss_weight=loss_cls_weight *
                      (num_classes / 80 * 3 / num_det_layers)),
        loss_obj=dict(loss_weight=loss_obj_weight *
                      ((img_scale[0] / 640)**2 * 3 / num_det_layers))))

pre_transform = _base_.pre_transform
albu_train_transforms = _base_.albu_train_transforms

mosaic_affine_pipeline = [
    dict(
        type='Mosaic',
        img_scale=img_scale,
        pad_val=114.0,
        pre_transform=pre_transform),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        # img_scale is (width, height)
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        border_val=(114, 114, 114))
]

# enable mixup
train_pipeline = [
    *pre_transform, *mosaic_affine_pipeline,
    dict(
        type='YOLOv5MixUp',
        prob=mixup_prob,
        pre_transform=[*pre_transform, *mosaic_affine_pipeline]),
    dict(
        type='mmdet.Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        }),
    dict(type='YOLOv5HSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction'))
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline,
                                     data_root=data_root,
                                     metainfo=metainfo,
                                     ann_file=train_ann_file,
                                     data_prefix=dict(img=train_data_prefix),),
                        batch_size=train_batch_size_per_gpu)
val_dataloader = dict(dataset=dict(data_root=data_root,
                                     metainfo=metainfo,
                                     data_prefix=dict(img=val_data_prefix),
                                     ann_file=val_ann_file,))
test_dataloader = val_dataloader
val_evaluator = dict(ann_file=data_root + val_ann_file)
test_evaluator = val_evaluator
default_hooks = dict(param_scheduler=dict(lr_factor=lr_factor,
                                          max_epochs=max_epochs))
train_cfg = dict(max_epochs=max_epochs)
visualizer = dict(vis_backends=[dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')])