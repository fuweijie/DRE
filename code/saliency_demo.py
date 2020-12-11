import torch
from sal.utils.pytorch_fixes import *
from sal.utils.pytorch_trainer import *
from sal.saliency_model import SaliencyModel, SaliencyLoss, get_black_box_fn,  Distribution_Controller
from sal.visual_masks import *
from sal.datasets import imagenet_dataset
from sal.utils.resnet_encoder import resnet50encoder
from torchvision.models.resnet import resnet50
from  torch.optim import lr_scheduler
#import pycat
import torch.nn as nn
import matplotlib.pyplot as plt
#from PIL import Image
import copy
import time
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
dts = imagenet_dataset
black_box_fn = get_black_box_fn(model_zoo_model=resnet50)
val_dts = dts.get_val_dataset()
allow_selector = True

' ---------------------------------------------------Testing with Mask Estimator -------------------------------------------'
batch_size=2
val_datas = dts.get_loader(val_dts, batch_size=batch_size,  Shuffle=True)
saliency = SaliencyModel(resnet50encoder(pretrained=True, require_out_grad=False), 5, 64, 3, 64, fix_encoder=True, use_simple_activation=False, allow_selector=allow_selector, num_classes=1000)
load_path='../data/yoursaliencymodel'
saliency.minimialistic_restore(os.path.join(os.path.dirname(__file__), (load_path)))
saliency.train(False)
saliency_p = saliency.to(device)

for it_step, batch in enumerate(val_datas):
    images, _, paths = batch
    images=images.to(device)
    outputs = saliency.encoder(images)
    feature_conv5=outputs[5]
    logits=outputs[-1]
    _, targets=torch.max(logits,dim=1)
    with torch.no_grad():
        outputs = saliency_p(images)
    proposed_pre_masks= outputs[0]
    proposed_pre_masks = F.upsample(proposed_pre_masks, (images.size(2), images.size(3)), mode='bilinear')
    proposed_post_masks=correct_masks_adversal(black_box_fn, images, proposed_pre_masks, targets, num_bins=8)
    proposed_pre_masks=proposed_pre_masks.squeeze(dim=1).cpu()

    for it_number in range((batch_size)):
        this_image=images[it_number].cpu().numpy()
        this_image = np.transpose(this_image, (1, 2, 0))
        mean=np.array([0.5,0.5,0.5])
        std=np.array([0.5,0.5,0.5])
        this_image=this_image*std+mean
        this_label=targets[it_number]
        save_heatmaps(this_image, this_label, proposed_pre_masks[it_number], proposed_post_masks[it_number], it_number)

