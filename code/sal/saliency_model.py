from sal.utils.pytorch_fixes import *
from torch.nn import functional as F
import torch.nn as nn
from torch.nn import Module
from sal.utils.mask import *
from torchvision.models.resnet import resnet50
import os
import numpy as np
import copy
import torch
from torchvision.models.resnet import ResNet

class Distribution_Controller(Module):
    def __init__(self):
        super(Distribution_Controller, self).__init__()
        self.conv7=nn.Conv2d(in_channels=64, out_channels=1, kernel_size=9, stride=8, padding=4)#-----#
        self.conv14=nn.Conv2d(in_channels=64, out_channels=1, kernel_size=5, stride=4, padding=2)#-----#
        self.conv28=nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=2, padding=1)#-----#
        self.conv56=nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.in7=nn.InstanceNorm2d(1)#-----#
        self.in14=nn.InstanceNorm2d(1)#-----#
        self.in28=nn.InstanceNorm2d(1)#-----#
        self.in56=nn.InstanceNorm2d(1)
        self.avepool=nn.AvgPool2d(3,1,padding=1)

    def forward(self, inputs):
        outputs=self.conv56(inputs)
        outputs=self.in56(outputs)
        outputs=self.transform(outputs)
        outputs=F.upsample(outputs,(56,56),mode='bilinear')
        return outputs
    
    def transform(self, inputs):
        outputs=torch.sigmoid(inputs*1.5)
        outputs=outputs**2.5
        return outputs

def get_black_box_fn(model_zoo_model=resnet50, image_domain=(-2., 2.)):
    black_box_model =  model_zoo_model(pretrained=True)
    black_box_model.eval()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    black_box_model = (black_box_model).to(device)
    def black_box_fn(_images):
        return black_box_model(_images)#adapt _to_image_domain(_images, image_domain))
    return black_box_fn

class SaliencyModel(Module):
    def __init__(self, encoder, encoder_scales, encoder_base, upsampler_scales, upsampler_base, fix_encoder=True, feed_grad=False,
                 use_simple_activation=False, allow_selector=False, selector_module=None, num_classes=1000):
        super(SaliencyModel, self).__init__()
        assert upsampler_scales <= encoder_scales

        encoder.eval()
        self.encoder = encoder  # decoder must return at least scale0 to scaleN where N is num_scales
        self.upsampler_scales = upsampler_scales
        self.encoder_scales = encoder_scales
        self.fix_encoder = fix_encoder
        self.use_simple_activation = use_simple_activation
        self.num_classes=num_classes
        down = self.encoder_scales
        modulator_sizes = []
        for up in reversed(range(self.upsampler_scales)):
            upsampler_chans = upsampler_base * 2**(up+1)
            encoder_chans = encoder_base * 2**down
            inc = upsampler_chans if down!=encoder_scales else encoder_chans
            modulator_sizes.append(inc)
            self.add_module('up%d'%up,
                            BottleneckCell(
                                in_channels=inc,
                                passthrough_channels=int(encoder_chans/2),
                                out_channels=int(upsampler_chans/2),
                                follow_up_residual_blocks=1,
                                activation_fn=lambda: nn.ReLU(),
                            ))
            down -= 1
        self.to_saliency_chans = Distribution_Controller() 
        self.batchnorm_op = nn.BatchNorm2d(1)#-----#
        self.relu = nn.ReLU(inplace=True)#-----#

        self.feed_grad=feed_grad
        self.allow_selector = allow_selector
        s = encoder_base*2**encoder_scales
        self.selector_module = nn.Embedding(num_classes, s)
        self.selector_module.weight.data.normal_(0, 1./s**0.5)
        
    def minimialistic_restore(self, save_dir):
        assert self.fix_encoder, 'You should not use this function if you are not using a pre-trained encoder like resnet'

        p = os.path.join(save_dir, 'model-%d.ckpt' % 1)
        if not os.path.exists(p):
            print('Could not find any checkpoint at %s, skipping restore' % p)
            return
        for name, data in torch.load(p, map_location=lambda storage, loc: storage).items():
            self._modules[name].load_state_dict(data)

    def minimalistic_save(self, save_dir):
        assert self.fix_encoder, 'You should not use this function if you are not using a pre-trained encoder like resnet'
        data = {}
        for name, module in self._modules.items():
            if module is self.encoder:  # we do not want to restore the encoder as it should have its own restore function
                continue
            data[name] = module.state_dict()
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        torch.save(data, os.path.join(save_dir, 'model-%d.ckpt' % 1))

    def get_trainable_parameters(self):
        all_params = self.parameters()
        if not self.fix_encoder: return set(all_params)
        unwanted = self.encoder.parameters()
        return set(all_params) - set(unwanted) - (set(self.selector_module.parameters()) if self.allow_selector else set([]))

    def forward(self,  _images,step=2000):     
        with torch.no_grad():
            out = self.encoder(_images)

        if self.fix_encoder:
            out = [e.detach() for e in out]
        down = self.encoder_scales
        main_flow = out[down]
  
        logit=out[-1]
        _, targets=torch.max(logit,1)

        em = torch.squeeze(self.selector_module(targets.view(-1, 1)), 1)
        act = torch.sum(main_flow*em.view(-1, 2048, 1, 1), 1, keepdim=True)
        abstract_masks = torch.sigmoid(act)
        abstract_masks = abstract_masks.repeat(1,2048,1,1)
        main_flow = main_flow.mul(abstract_masks)
        #ex = torch.mean(torch.mean(act, 3), 2)
        #exists_logits = torch.cat((-ex / 2., ex / 2.), 1)

        for up in reversed(range(self.upsampler_scales)):
            assert down > 0
            main_flow = self._modules['up%d'%up](main_flow, out[down-1])
            down -= 1

        mask = self.to_saliency_chans(main_flow)
        return mask, targets

class SaliencyLoss():
    def __init__(self, black_box_fn, **apply_mask_kwargs):
        
        self.apply_mask_kwargs = apply_mask_kwargs
        self.black_box_fn = black_box_fn
        
    def get_loss(self, _images, _targets, _masks):
        if _masks.size()[-2:] != _images.size()[-2:]:
            _masks = F.upsample(_masks, (_images.size(2), _images.size(3)), mode='bilinear')
        preserved_images = apply_mask(_images, _masks)
        preserved_logits = self.black_box_fn(preserved_images)
        total_loss=nn.CrossEntropyLoss()(preserved_logits,_targets)

        return total_loss
  
