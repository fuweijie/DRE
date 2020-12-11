import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
from PIL import Image
import copy
import matplotlib.image as mpimg
from sal.process_bounding_boxes import ProcessXMLAnnotation
from skimage.transform import rescale, resize
import scipy
import torch
from sal.utils.mask import apply_mask, gaussian_blur
from torch.nn import functional as F
from sklearn import preprocessing
from sklearn.metrics import average_precision_score
from scipy.ndimage import gaussian_filter

def save_heatmaps(this_image, this_label, proposed_pre_masks, proposed_post_masks, it_number): 
        ax1=plt.subplot(1,3,1); ax1.set_title('Img:%d.' % (this_label),fontsize=3); ax1.imshow(this_image);ax1.axis('off')
        ax2=plt.subplot(1,3,2); ax2.set_title('Pre_mask'); ax2.imshow(proposed_pre_masks, cmap ='jet'); ax2.axis('off')
        ax3=plt.subplot(1,3,3); ax3.set_title('Post_mask'); ax3.imshow(np.multiply(proposed_post_masks,proposed_pre_masks), cmap ='jet'); ax3.axis('off')
        plt.axis('off');
        plt.savefig('../results/eval_my_image_%d.jpg' % (it_number), facecolor='gray', bbox_inches='tight', dpi=300)
        plt.close() 

def correct_masks_adversal(black_box_fn, images, im_masks, targets, num_bins=32):
    these_images=images.repeat(num_bins,1,1,1)
    these_targets=targets.view(-1,1).repeat(num_bins,1).view(-1)
    ratio_line=np.linspace(0,1,num_bins)
    delta_x=1/num_bins
    num_images=images.size()[0]
    after_masks=np.zeros((num_images, 224,224))
    top_hot_kept_mask, selected_mask_weight =keep_top_k_units(im_masks,ratio_line,num_bins)
    n, c, h, w = im_masks.size()
    cand = gaussian_blur(these_images, sigma=10)
    masked_images = these_images.mul(top_hot_kept_mask)+(1-top_hot_kept_mask).mul(cand)
    with torch.no_grad():
        logits=black_box_fn(masked_images)
    soft_logits=torch.softmax(logits, dim=1)
    these_soft_logits=soft_logits[torch.arange(num_bins*num_images),these_targets].reshape(-1,num_images).transpose(1,0)
    soft_acc_masks=these_soft_logits.detach().cpu().numpy()

    for it_image in range(num_images):
        temp_soft_acc_masks=copy.deepcopy(these_soft_logits[it_image,:]).cpu().numpy()
        temp_mask=copy.deepcopy(im_masks[it_image,:][0]).cpu().numpy()
        temp_image=these_images[0]
        temp_image=np.transpose(temp_image.cpu().numpy(),(1,2,0))
        mean=np.array([0.5,0.5,0.5])
        std=np.array([0.5,0.5,0.5])
        temp_image=temp_image*std+mean
        temp_val=copy.deepcopy(temp_soft_acc_masks)
        for it_correct in np.arange(num_bins-1):
            size_area=images.size()[2]*images.size()[3]
            temp_val[it_correct+1]=np.maximum(temp_val[it_correct],temp_val[it_correct+1])
        topk=np.maximum(np.floor(size_area*ratio_line)-1,0).astype(int)
        inter_x=np.arange(size_area)
        vals=(np.max(temp_soft_acc_masks)-temp_val)/(np.max(temp_soft_acc_masks)-np.min(temp_soft_acc_masks))
        vals=np.maximum(vals,0)
        vals = np.interp(inter_x, topk, vals[:])
        sort_mask_ind=np.argsort(-temp_mask.reshape(-1))
        temp_mask.reshape(-1)[sort_mask_ind]=vals+1e-15
        after_masks[it_image,:]=temp_mask
    return after_masks

def keep_top_k_units(image_masks,ratio_line,num_bins):
    num_images=image_masks.size()[0]
    size_area=(image_masks.size()[-1])*(image_masks.size()[-2])
    topk=np.maximum(np.floor(size_area*ratio_line)-1,0).astype(int).repeat(num_images)
    pre_image_masks=copy.deepcopy(image_masks)
    post_image_masks=pre_image_masks.view(num_images,-1)
    temp_val, temp_ind = torch.sort(post_image_masks, descending=True, dim=1)
    temp_val=temp_val[torch.arange(num_images).repeat(num_bins),topk]
    these_im_masks=pre_image_masks.repeat(num_bins,1,1,1)
    top_hot_kept_mask=these_im_masks-temp_val.unsqueeze(dim=1).unsqueeze(dim=2).unsqueeze(dim=3).repeat(1,1,(image_masks.size()[-1]),(image_masks.size()[-2]))
    top_hot_kept_mask=F.relu(top_hot_kept_mask)
    top_hot_kept_mask[top_hot_kept_mask>0]=1
    selected_mask_weight=torch.sum(top_hot_kept_mask.mul(these_im_masks),dim=[1,2,3])
    return top_hot_kept_mask, selected_mask_weight
