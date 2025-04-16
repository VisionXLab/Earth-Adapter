from mmengine.runner import Runner
from mmengine.registry import RUNNERS
from typing import Callable, Union
from mmengine.runner.checkpoint import _load_checkpoint,_load_checkpoint_to_model,find_latest_checkpoint
from mmengine.model import is_model_wrapper
@RUNNERS.register_module()
class custom_runner(Runner):

    def test(self) -> dict:
        """Launch test.

        Returns:
            dict: A dict of metrics on testing set.
        """
        if self._test_loop is None:
            raise RuntimeError(
                '`self._test_loop` should not be None when calling test '
                'method. Please provide `test_dataloader`, `test_cfg` and '
                '`test_evaluator` arguments when initializing runner.')

        self._test_loop = self.build_test_loop(self._test_loop)  # type: ignore

        self.call_hook('before_run')
        self.model.init_weights()

        # make sure checkpoint-related hooks are triggered after `before_run`
        self.load_or_resume()

        metrics = self.test_loop.run()  # type: ignore
        self.call_hook('after_run')
        return metrics
    def load_checkpoint(self,
                        filename: str,
                        map_location: Union[str, Callable] = 'cpu',
                        strict: bool = False,
                        revise_keys: list = [(r'^module.', '')]):
        """Load checkpoint from given ``filename``.

        Args:
            filename (str): Accept local filepath, URL, ``torchvision://xxx``,
                ``open-mmlab://xxx``.
            map_location (str or callable): A string or a callable function to
                specifying how to remap storage locations.
                Defaults to 'cpu'.
            strict (bool): strict (bool): Whether to allow different params for
                the model and checkpoint.
            revise_keys (list): A list of customized keywords to modify the
                state_dict in checkpoint. Each item is a (pattern, replacement)
                pair of the regular expression operations. Defaults to strip
                the prefix 'module.' by [(r'^module\\.', '')].
        """
        checkpoint = _load_checkpoint(filename, map_location=map_location)
        state_dict = checkpoint['state_dict']
        for key in list(state_dict.keys()):
            if key.startswith('ema_model.backbone'):
                del state_dict[key]
            elif key.startswith('ema_model.decode_head'):
                state_dict[key.replace('ema_model.decode_head', 'decode_head')] = state_dict.pop(key)

        # Add comments to describe the usage of `after_load_ckpt`
        self.call_hook('after_load_checkpoint', checkpoint=checkpoint)

        if is_model_wrapper(self.model):
            model = self.model.module
        else:
            model = self.model

        checkpoint = _load_checkpoint_to_model(
            model, checkpoint, strict, revise_keys=revise_keys)

        self._has_loaded = True

        self.logger.info(f'Load checkpoint from {filename}')

        return checkpoint