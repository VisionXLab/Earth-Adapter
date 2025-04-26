# dataset config
_base_ = [
    "../../../_base_/DG_dataset/dg_r2u.py",
    "../../../_base_/default_runtime.py",
    "../../../_base_/models/dinov2_mask2former.py"
]
model = dict(
    backbone = dict(
        type = 'MOE_Adpter_DinoVisionTransformer',
        moe_adapter_type = 'earth_adapter',
        adapter_config = dict(
            dim = 64,
            with_token = False,
            fft_layer = [0,1,2,3,4,5],
            cutoff_ratio = 0.3
        ),
    ),
    decode_head = dict(
    num_classes= 7,
    loss_cls = dict(
        class_weight = [1.0]*7+[0.1],
        )
    )
)
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
#TODO:
optim_wrapper = dict(
    constructor="PEFTOptimWrapperConstructor",
    optimizer=dict(
        type="AdamW", lr=0.0001, weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999)
    ),
    paramwise_cfg=dict(
        custom_keys={
            "norm": dict(decay_mult=0.0),
            "query_embed": embed_multi,
            "level_embed": embed_multi,
            "learnable_tokens": embed_multi,
            "reins.scale": embed_multi,
        },
        norm_decay_mult=0.0,
    ),
)
param_scheduler = [
    dict(type="PolyLR", eta_min=0, power=0.9, begin=0, end=20000, by_epoch=False)
]
train_cfg = dict(type="IterBasedTrainLoop", max_iters=20000, val_interval=2000)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2000,max_keep_ckpts=1,save_best='mIoU'),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="SegVisualizationHook",interval=1),
)
exp_name = 'DG'
randomness = dict(seed =0)
