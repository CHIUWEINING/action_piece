# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Trainer for ActionPiece.

This module defines the Trainer class, which handles the training process for an
ActionPiece model. It includes methods for fitting the model, evaluating it, and
managing resources.
"""

import collections
import logging
import os
from typing import Any

from genrec.evaluator import Evaluator
from genrec.model import AbstractModel
from genrec.tokenizer import AbstractTokenizer
from genrec.utils import config_for_ckpt
from genrec.utils import config_for_log
from genrec.utils import get_best_ckpt_path
from genrec.utils import get_file_name
from genrec.utils import get_last_ckpt_path
from genrec.utils import get_rng_states
from genrec.utils import get_total_steps
from genrec.utils import load_ckpt
from genrec.utils import log
from genrec.utils import set_rng_states
import numpy as np
import torch
from torch import optim
from torch.nn import utils
import tqdm
from transformers import optimization


get_scheduler = optimization.get_scheduler
tqdm = tqdm.tqdm
AdamW = optim.AdamW
clip_grad_norm_ = utils.clip_grad_norm_
getLogger = logging.getLogger
OrderedDict = collections.OrderedDict


class Trainer:
  """A class that handles the training process for a model.

  Attributes:
      config (dict): The configuration parameters for training.
      model (AbstractModel): The model to be trained.
      evaluator (Evaluator): The evaluator used for evaluating the model.
      logger (Logger): The logger used for logging training progress.
      project_dir (str): The directory path for saving tensorboard logs.
      saved_model_ckpt (str): The file path for saving the best model
        checkpoint (model weights only). With `test_only` it is `ckpt_path`.
      last_ckpt (str): The file path for saving the full training state (model,
        optimizer, scheduler, epoch, RNG states, ...) after every epoch. Pass it
        as `resume_from` to resume training.
      accelerator: The accelerator used for training.

  Methods:
      fit(train_dataloader, val_dataloader): Trains the model using the provided
        training and validation dataloaders.
      evaluate(dataloader, split='test'): Evaluate the model on the given
        dataloader.
      end(): Ends the training process and releases any used resources.
  """

  def __init__(self, config: dict[Any, Any], model: AbstractModel,
               tokenizer: AbstractTokenizer):
    """Initializes the Trainer with the given configuration, model, and tokenizer.

    Args:
        config (dict): The configuration parameters for training.
        model (AbstractModel): The model to be trained.
        tokenizer (AbstractTokenizer): The tokenizer used for tokenizing the
          data.
    """
    self.config = config
    self.model = model
    self.accelerator = config['accelerator']
    self.evaluator = Evaluator(config, tokenizer)
    self.logger = getLogger()

    self.resume_from = None
    if self.config['test_only']:
      # Pure testing: evaluate an existing checkpoint, nothing is written to
      # the ckpt dir.
      ckpt_path = self.config['ckpt_path']
      if not ckpt_path or not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f'test_only requires an existing --ckpt_path, got: {ckpt_path}'
        )
      category = self.config['category']
      if os.path.basename(os.path.dirname(ckpt_path)) != category:
        self.log(
            f'Checkpoint {ckpt_path} is not under a "{category}" folder,'
            ' please double check --category.',
            level='warning',
        )
      self.saved_model_ckpt = ckpt_path
      self.last_ckpt = None
      return

    if self.config['resume_from']:
      # Keep writing to the checkpoints of the run being resumed.
      self.resume_from = get_last_ckpt_path(self.config['resume_from'])
      if not os.path.isfile(self.resume_from):
        raise FileNotFoundError(
            f'Cannot resume: {self.resume_from} not found (resume_from must be'
            ' a .last.pth checkpoint or the .pth next to it).'
        )
      self.saved_model_ckpt = get_best_ckpt_path(self.resume_from)
    else:
      self.saved_model_ckpt = os.path.join(
          self.config['ckpt_dir'],
          self.config['category'],
          get_file_name(self.config, suffix='.pth'),
      )
    self.last_ckpt = get_last_ckpt_path(self.saved_model_ckpt)
    os.makedirs(os.path.dirname(self.saved_model_ckpt), exist_ok=True)

  def fit(self, train_dataloader, val_dataloader):
    """Trains the model using the provided training and validation dataloaders.

    Args:
        train_dataloader: The dataloader for training data.
        val_dataloader: The dataloader for validation data.
    """
    optimizer = AdamW(
        self.model.parameters(),
        lr=self.config['lr'],
        weight_decay=self.config['weight_decay'],
    )

    total_n_steps = get_total_steps(self.config, train_dataloader)
    if total_n_steps == 0:
      self.log('No training steps needed.')
      return

    scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=self.config['warmup_steps'],
        num_training_steps=total_n_steps,
    )

    self.model, optimizer, train_dataloader, val_dataloader, scheduler = (
        self.accelerator.prepare(
            self.model, optimizer, train_dataloader, val_dataloader, scheduler
        )
    )
    self.accelerator.init_trackers(
        project_name=get_file_name(self.config, suffix=''),
        config=config_for_log(self.config),
        init_kwargs={'tensorboard': {'flush_secs': 60}},
    )

    n_epochs = np.ceil(
        total_n_steps / (len(train_dataloader) * self.accelerator.num_processes)
    ).astype(int)
    start_epoch = 0
    best_epoch = 0
    best_val_score = -1
    if self.resume_from:
      start_epoch, best_epoch, best_val_score = self._load_training_state(
          optimizer, scheduler
      )
      if self._should_early_stop(start_epoch, best_epoch):
        self.log(f'Early stopping already triggered at epoch {start_epoch}')
        n_epochs = start_epoch

    for epoch in range(start_epoch, n_epochs):
      # Training
      self.model.train()
      total_loss = 0.0
      train_progress_bar = tqdm(
          train_dataloader,
          total=len(train_dataloader),
          desc=f'Training - [Epoch {epoch + 1}]',
      )
      for batch in train_progress_bar:
        optimizer.zero_grad()
        outputs = self.model(batch)
        loss = outputs.loss
        self.accelerator.backward(loss)
        if self.config['max_grad_norm'] is not None:
          clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])
        optimizer.step()
        scheduler.step()
        total_loss = total_loss + loss.item()

      self.accelerator.log(
          {'Loss/train_loss': total_loss / len(train_dataloader)},
          step=epoch + 1,
      )
      self.log(
          f'[Epoch {epoch + 1}] Train Loss:'
          f' {total_loss / len(train_dataloader)}'
      )

      # Evaluation
      early_stop = False
      if (epoch + 1) % self.config['eval_interval'] == 0:
        all_results = self.evaluate(val_dataloader, split='val')
        if self.accelerator.is_main_process:
          for key in all_results:
            self.accelerator.log(
                {f'Val_Metric/{key}': all_results[key]}, step=epoch + 1
            )
          self.log(f'[Epoch {epoch + 1}] Val Results: {all_results}')

        val_score = all_results[self.config['val_metric']]
        if val_score > best_val_score:
          best_val_score = val_score
          best_epoch = epoch + 1
          if self.accelerator.is_main_process:
            self._atomic_save(
                self._unwrapped_state_dict(), self.saved_model_ckpt
            )
            self.log(
                f'[Epoch {epoch + 1}] Saved model checkpoint to'
                f' {self.saved_model_ckpt}'
            )

        early_stop = self._should_early_stop(epoch + 1, best_epoch)

      # The training state is saved after every epoch (also when early
      # stopping), so that the run can be resumed from the latest epoch.
      if self.accelerator.is_main_process:
        self._save_training_state(
            optimizer, scheduler, epoch + 1, best_epoch, best_val_score
        )
      if early_stop:
        self.log(f'Early stopping at epoch {epoch + 1}')
        break

    self.log(f'Best epoch: {best_epoch}, Best val score: {best_val_score}')

  def _should_early_stop(self, n_epochs_done: int, best_epoch: int) -> bool:
    """Whether early stopping is triggered after `n_epochs_done` epochs."""
    return (
        self.config['patience'] is not None
        and n_epochs_done > 0
        and n_epochs_done % self.config['eval_interval'] == 0
        and n_epochs_done - best_epoch >= self.config['patience']
    )

  def _unwrapped_state_dict(self):
    return self.accelerator.unwrap_model(self.model).state_dict()

  @staticmethod
  def _atomic_save(obj, path):
    """Saves via a temp file, so an interrupted save cannot corrupt `path`."""
    tmp_path = path + '.tmp'
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)

  def _save_training_state(
      self, optimizer, scheduler, epoch, best_epoch, best_val_score
  ):
    """Saves everything needed to resume training to `self.last_ckpt`."""
    self._atomic_save(
        {
            'model': self._unwrapped_state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,  # number of finished epochs
            'best_epoch': best_epoch,
            'best_val_score': best_val_score,
            'rng_states': get_rng_states(),
            'config': config_for_ckpt(self.config),
        },
        self.last_ckpt,
    )

  def _load_training_state(self, optimizer, scheduler):
    """Restores the training state saved by `_save_training_state`.

    Args:
        optimizer: The (prepared) optimizer.
        scheduler: The (prepared) scheduler.

    Returns:
        (start_epoch, best_epoch, best_val_score)
    """
    state = load_ckpt(self.resume_from)
    if 'optimizer' not in state:
      raise ValueError(
          f'{self.resume_from} is not a training-state checkpoint (.last.pth).'
      )
    self.accelerator.unwrap_model(self.model).load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    set_rng_states(state['rng_states'])

    ignored_keys = {
        'run_local_time', 'resume_from', 'ckpt_path', 'test_only', 'results_dir'
    }
    cur_config = config_for_ckpt(self.config)
    diff = sorted(
        key
        for key in (state['config'].keys() | cur_config.keys()) - ignored_keys
        if state['config'].get(key) != cur_config.get(key)
    )
    if diff:
      self.log(
          'Config differs from the checkpointed run in: '
          + ', '.join(
              f'{k} ({state["config"].get(k)} -> {cur_config.get(k)})'
              for k in diff
          ),
          level='warning',
      )
    self.log(
        f'Resumed from {self.resume_from} at epoch {state["epoch"]} (best'
        f' epoch: {state["best_epoch"]}, best val score:'
        f' {state["best_val_score"]})'
    )
    return state['epoch'], state['best_epoch'], state['best_val_score']

  def evaluate(self, dataloader, split='test'):
    """Evaluates the model on the given dataloader.

    Args:
        dataloader (torch.utils.data.DataLoader): The dataloader to evaluate on.
        split (str, optional): The split name. Defaults to 'test'.

    Returns:
        collections.OrderedDict: A dictionary containing the evaluation results.
    """
    self.model.eval()

    all_results = collections.defaultdict(list)
    val_progress_bar = tqdm(
        dataloader,
        total=len(dataloader),
        desc=f'Eval - {split}',
    )
    for batch in val_progress_bar:
      with torch.no_grad():
        batch = {k: v.to(self.accelerator.device) for k, v in batch.items()}
        if self.config[
            'use_ddp'
        ]:  # ddp, gather data from all devices for evaluation
          preds = self.model.module.generate(
              batch, n_return_sequences=self.evaluator.maxk
          )
          all_preds, all_labels = self.accelerator.gather_for_metrics(
              (preds, batch['labels'])
          )
          results = self.evaluator.calculate_metrics(all_preds, all_labels)
        else:
          preds = self.model.generate(
              batch, n_return_sequences=self.evaluator.maxk
          )
          results = self.evaluator.calculate_metrics(preds, batch['labels'])
        for key, value in results.items():
          all_results[key].append(value)

    output_results = OrderedDict()
    for metric in self.config['metrics']:
      for k in self.config['topk']:
        key = f'{metric}@{k}'
        output_results[key] = torch.cat(all_results[key]).mean().item()
    return output_results

  def end(self):
    """Ends the training process and releases any used resources."""
    self.accelerator.end_training()

  def log(self, message, level='info'):
    return log(message, self.config['accelerator'], self.logger, level=level)
