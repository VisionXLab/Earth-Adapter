from typing import Dict,Union
import torch
from mmengine.optim import OptimWrapper
from mmseg.models.segmentors.base import BaseSegmentor
from torch import Tensor
from mmseg.utils import OptSampleList,ForwardResults

class custom_BaseSegmentor(BaseSegmentor):
    def _run_forward(self, data: Union[dict, tuple, list],
                     mode: str,iter = None) -> Union[Dict[str, torch.Tensor], list]:
        """Unpacks data for :meth:`forward`

        Args:
            data (dict or tuple or list): Data sampled from dataset.
            mode (str): Mode of forward.

        Returns:
            dict or list: Results of training or testing mode.
        """
        if isinstance(data, dict):
            results = self(**data, mode=mode,iter = iter)
        elif isinstance(data, (list, tuple)):
            results = self(*data, mode=mode,iter = iter)
        else:
            raise TypeError('Output of `data_preprocessor` should be '
                            f'list, tuple or dict, but got {type(data)}')
        return results
    def train_step(self, data: Union[dict, tuple, list],
                   optim_wrapper: OptimWrapper,iter = None) -> Dict[str, torch.Tensor]:
        # Enable automatic mixed precision training context.
        with optim_wrapper.optim_context(self):
            data = self.data_preprocessor(data, True)
            losses = self._run_forward(data, mode='loss',iter = iter)  # type: ignore
        parsed_losses, log_vars = self.parse_losses(losses)  # type: ignore
        optim_wrapper.update_params(parsed_losses)
        return log_vars
    def forward(self,
                inputs: Tensor,
                data_samples: OptSampleList = None,
                mode: str = 'tensor',iter = None) -> ForwardResults:
        if mode == 'loss':
            return self.loss(inputs, data_samples,iter = iter)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')    