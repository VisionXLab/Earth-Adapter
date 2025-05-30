from mmseg.models.builder import BACKBONES, MODELS
from .dino_v2 import DinoVisionTransformer
from .utils import set_requires_grad, set_train
from .earth_adapter import earth_adapter
moe_adapter = {
    "earth_adapter": earth_adapter,
}
@BACKBONES.register_module()
class MOE_Adpter_DinoVisionTransformer(DinoVisionTransformer):
    def __init__(
        self,
        moe_adapter_type = None,
        adapter_config = {},
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.refine_feat = moe_adapter[moe_adapter_type](**adapter_config)
    def forward_features(self, x, masks=None):
        B, _, h, w = x.shape
        H, W = h // self.patch_size, w // self.patch_size
        x = self.prepare_tokens_with_masks(x, masks)
        outs = []
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            x = self.refine_feat.forward(#
                x,#4,1025,1024
                idx,
                batch_first=True,
                has_cls_token=True,
            )
            if idx in self.out_indices:
                outs.append(
                    x[:, 1:, :].permute(0, 2, 1).reshape(B, -1, H, W).contiguous()
                )
        return outs
    def train(self, mode: bool = True):
        if not mode:
            return super().train(mode)#评估模式
        set_requires_grad(self, ["refine_feat"])
        set_train(self, ["refine_feat"])
    # def state_dict(self, destination, prefix, keep_vars):
    #     state = super().state_dict(destination, prefix, keep_vars)
    #     keys = [k for k in state.keys() if "refine_feat" not in k]
    #     # keys.extend[[k for k in state.keys() if "a_ema_model" in k]]
    #     for key in keys:
    #         state.pop(key)
    #         if key in destination:
    #             destination.pop(key)
    #     return state
