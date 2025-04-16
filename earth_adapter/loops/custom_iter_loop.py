from mmengine.runner.loops import IterBasedTrainLoop,_InfiniteDataloaderIterator,calc_dynamic_intervals
from mmengine.registry import LOOPS
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Sequence, Tuple, Union
from mmengine.logging import HistoryBuffer, print_log
import logging

@LOOPS.register_module()
class my_iter_loop(IterBasedTrainLoop):
    def __init__(
                self,
                runner,
                dataloader: Union[DataLoader, Dict],
                max_iters: int,
                val_begin: int = 1,
                val_interval: int = 1000,
                val_before_train = False,
                dynamic_intervals: Optional[List[Tuple[int, int]]] = None) -> None:
            super().__init__(runner, dataloader,max_iters,val_begin,val_interval,dynamic_intervals)
            self.val_before_train = val_before_train
    def run(self) -> None:
            """Launch training."""
            self.runner.call_hook('before_train')
            # In iteration-based training loop, we treat the whole training process
            # as a big epoch and execute the corresponding hook.
            self.runner.call_hook('before_train_epoch')
            if self.val_before_train:
                self.runner.val_loop.run()
            if self._iter > 0:
                print_log(
                    f'Advance dataloader {self._iter} steps to skip data '
                    'that has already been trained',
                    logger='current',
                    level=logging.WARNING)
                for _ in range(self._iter):
                    next(self.dataloader_iterator)
            while self._iter < self._max_iters and not self.stop_training:
                self.runner.model.train()
                data_batch = next(self.dataloader_iterator)
                self.run_iter(data_batch)
                self._decide_current_val_interval()
                if (self.runner.val_loop is not None
                        and self._iter >= self.val_begin
                        and (self._iter % self.val_interval == 0
                            or self._iter == self._max_iters)):
                    self.runner.val_loop.run()
            self.runner.call_hook('after_train_epoch')
            self.runner.call_hook('after_train')
            return self.runner.model

    def run_iter(self, data_batch: Sequence[dict]) -> None:
        """Iterate one mini-batch.
        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook(
            'before_train_iter', batch_idx=self._iter, data_batch=data_batch)
        # Enable gradient accumulation mode and avoid unnecessary gradient
        # synchronization during gradient accumulation process.
        # outputs should be a dict of loss.
        outputs = self.runner.model.train_step(
            data_batch, optim_wrapper=self.runner.optim_wrapper,iter=self._iter)
        self.runner.call_hook(
            'after_train_iter',
            batch_idx=self._iter,
            data_batch=data_batch,
            outputs=outputs)
        self._iter += 1
    