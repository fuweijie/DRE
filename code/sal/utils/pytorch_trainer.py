
import time
import sys
import torch
import numpy as np
import random
import os
import torch.nn
import torch.utils.data as torch_data
import torch.optim as torch_optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.modules.loss as losses
import matplotlib.pyplot as plt
from torch.autograd import Variable
#import pycat
import torch.nn.functional as F
import scipy.ndimage
plt.switch_backend('agg')
GREEN_STR = '\033[38;5;2m%s\033[0m'
RED_STR = '\033[38;5;1m%s\033[0m'
INFO_TEMPLATE = '\033[38;5;2mINFO: %s\033[0m\n'
WARN_TEMPLATE = '\033[38;5;1mWARNING: %s\033[0m\n'

#assert torch.cuda.is_available(), 'CUDA must be available'

from sal.utils.pt_store import PTStore, to_number, to_numpy
PT = PTStore()
BATCHES_DONE_INFO = '{batches_done}/{batches_per_epoch}'
TIME_INFO = 'time: {comp_time:.3f} - data: {data_time:.3f} - ETA: {eta:.0f}'
SPEED_INFO = 'e/s: {examples_per_sec:.1f}'

def smoothing_dict_update(main, update, smooth_coef):
    for k, v in update.items():
        if main.get(k) is None:
            main[k] = v
        else:
            main[k] = smooth_coef*main[k] + (1.-smooth_coef)*v
    return main

class NiceTrainer:
    def __init__(self,
                 forward_step,  # forward step takes the output of the transform_inputs
                 scheduler_lr,
                 train_dts,
                 optimizer,
                 pt_store=PT,
                 activation_mode=None,
                 fake_prob=0,
                 mask_loss_coff=None,
                 weight_loss_coff=None,
                 transform_inputs=lambda batch, trainer: batch,

                 printable_vars=(),
                 events=(),
                 computed_variables=None,

                 loss_name='loss',

                 val_dts=None,
                 set_trainable=None,

                 modules=None,
                 save_every=None,
                 save_dir='ntsave',

                 info_string=(BATCHES_DONE_INFO, TIME_INFO, SPEED_INFO),
                 smooth_coef=0.95,
                 goodbye_after = 5,

                 lr_step_period=None,
                 lr_step_gamma=0.1,
                 ):
        '''

        '''
        self.forward_step = forward_step
        assert isinstance(train_dts, torch_data.DataLoader),  'train_dts must be an instance of torch.utils.data.DataLoader'
        self.train_dts = train_dts
        assert isinstance(optimizer, torch_optim.Optimizer), 'optimizer must be an instance of torch.optim.Optimizer'
        self.optimizer = optimizer
        self.scheduler = scheduler_lr
        assert isinstance(pt_store, PTStore), 'pt_store must be an instance of PTStore'
        self.pt_store = pt_store
        
        self.transform_inputs = transform_inputs

        self.printable_vars = list(printable_vars)
        self.events = events #list(events) if events else []
        #assert all(map(lambda x: isinstance(x, BaseEvent), self.events)), 'All events must be instances of the BaseEvent!'
        self.computed_variables = computed_variables if computed_variables is not None else {}

        self.loss_name = loss_name

        assert val_dts is None or isinstance(val_dts, torch_data.DataLoader),  'val_dts must be an instance of torch.utils.data.DataLoader or None'
        self.val_dts = val_dts
        if modules is not None:
            if not hasattr(modules, '__iter__'):
                modules = [modules]
        assert modules is None or all(map(lambda x: isinstance(x, torch.nn.Module), modules)), 'The list of modules can only contain instances of torch.nn.Module'
        self.modules = modules
        if set_trainable is None and self.modules is not None:
            def set_trainable(is_training):
                for m in self.modules:
                    m.train(is_training)
        self.set_trainable = set_trainable

        self.save_every = save_every
        self.save_dir = save_dir

        self._is_in_train_mode = None

    def _main_loop(self, is_training, epochs=1, steps=None, allow_switch_mode=True):
        """Trains for 1 epoch if steps is None. Otherwise performs specified number of steps."""
        if steps is  not None: print(  WARN_TEMPLATE % 'Num steps is not fully supported yet! (fix it!)')  # todo allow continue and partial execution
        if not is_training:
            assert self.val_dts is not None, 'Validation dataset was not provided'
        if allow_switch_mode and self._is_in_train_mode != is_training:
            if self.set_trainable is not None:
                self.set_trainable(is_training=is_training)
                self._is_in_train_mode = is_training
            else:
                if is_training:
                    print (WARN_TEMPLATE % "could not set the modules to the training mode because neither set_trainable nor modules were provided, assuming already in the training mode")
                    self._is_in_train_mode = True
                else:
                    raise ValueError("cannot set the modules to the eval mode because neither set_trainable nor modules were provided")

        dts = self.train_dts if is_training else self.val_dts


        batches_per_epoch = len(dts)
        batch_size = dts.batch_size
        steps_done_here = 0
        batches_done = 0  # todo allow continue!

        for epoch in range(epochs):
            self.scheduler.step()
            print('-'*40)
            print('epoch: %d.' % (epoch))
            print('-'*40)
            step_account = 1
            for it_step, batch in enumerate(dts):
                print('epoch: %d, step: %d.' % (epoch,it_step))
                self.pt_store.clear()
                batch = self.transform_inputs(batch, self)
                self.pt_store.batch = batch
                self.pt_store.step=it_step
                # --------------------------- OPTIMIZATION STEP ------------------------------------
                if is_training:
                    self.optimizer.zero_grad()
                self.forward_step(batch)
                loss = getattr(self.pt_store, self.loss_name)

                if is_training:
                    loss.backward()
                    self.optimizer.step()

                step_account += 1
                if step_account >2500:
                    break

    def train(self, epochs=1, steps=None):
        if steps is None:
            print( '_'*55)
            #print('Epoch', self.info_vars['epochs_done']+1)
        self._main_loop(is_training=True, epochs=epochs, steps=steps, allow_switch_mode=True)

    def validate(self, epochs=1, allow_switch_mode=False):
        #old_info = self.info_vars.copy()
        print( "Validation:")
        self._main_loop(is_training=False, epochs=epochs,  steps=None, allow_switch_mode=allow_switch_mode)
        #self.info_vars = old_info

    def _get_state(self):
        return dict(
            info_vars={k:v for k, v in self.info_vars.items() if k in self._core_info_vars},
            state_dicts=[m.state_dict() for m in self.modules],
            optimizer_state=self.optimizer.state_dict(),
        )

    def _set_state(self, state):
        self.info_vars = state['info_vars']
        self.optimizer.load_state_dict(state['optimizer_state'])
        if len(self.modules)!=len(state['state_dicts']):
            raise ValueError('The number of save dicts is different from number of models')
        for m, s in zip(self.modules, state['state_dicts']):
            m.load_state_dict(s)

    def save(self, step=1):
        if not self.modules:
            raise ValueError("nothing to save - the list of modules was not provided")
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        torch.save(self._get_state(), os.path.join(self.save_dir, 'model-%d.ckpt'%step))

    def restore(self, step=1):
        if not self.modules:
            raise ValueError("nothing to load - the list of modules was not provided")
        p = os.path.join(self.save_dir, 'model-%d.ckpt' % step)
        if not os.path.exists(p):
            return
        self._set_state(torch.load(p))

def ev_batch_to_images_labels(func):
    def f(batch):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _images, _labels, _paths = batch
        _paths = PT(paths=(_paths))
        _images = PT(images=Variable(_images)).to(device)
        _labels = PT(labels=Variable(_labels)).to(device)
        return func(_images, _labels)
    return f
