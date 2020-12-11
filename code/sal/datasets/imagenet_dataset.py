from torchvision.transforms import *
from torchvision.datasets import ImageFolder
from torch.utils.data import dataloader
import time, random
from ..utils.pytorch_fixes import *
import os
from sal.datasets.imagenet_synset import synset
# Images must be segregated in folders by class! So both train and val folders should contain 1000 folders, one for each class.
# PLEASE EDIT THESE 2 LINES:
IMAGE_NET_TRAIN_PATH = '../data/yourexamples/'
IMAGE_NET_VAL_PATH = '../data/yourexamples/'
#-----------------------------------------------------
SUGGESTED_BS = 128
NUM_CLASSES = 1000
SUGGESTED_EPOCHS_PER_STEP = 11
SUGGESTED_BASE = 64

STD_NORMALIZE = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#STD_NORMALIZE = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225])

class ImageFolderWithPaths(ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """
    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

def get_train_dataset(size=224):
    if not (os.path.exists(IMAGE_NET_TRAIN_PATH) and os.path.exists(IMAGE_NET_VAL_PATH)):
        raise ValueError(
            'Please make sure that you specify a path to the ImageNet dataset folder in sal/datasets/imagenet_dataset.py file!')
    return ImageFolderWithPaths(IMAGE_NET_TRAIN_PATH, transform=Compose([
        Resize(224), #RandomSizedCrop2(size, min_area=0.3),
        CenterCrop(size), #RandomHorizontalFlip(),
        ToTensor(),
        STD_NORMALIZE,  # Images will be in range -1 to 1
    ]))


def get_val_dataset(size=224):
    if not (os.path.exists(IMAGE_NET_TRAIN_PATH) and os.path.exists(IMAGE_NET_VAL_PATH)):
        raise ValueError(
            'Please make sure that you specify a path to the ImageNet dataset folder in sal/datasets/imagenet_dataset.py file!')
    return ImageFolderWithPaths(IMAGE_NET_VAL_PATH, transform=Compose([
        Resize(224),
        CenterCrop(size),
        ToTensor(),
        STD_NORMALIZE,
    ]))

def get_loader(dataset, batch_size=64, pin_memory=True, Shuffle=True):
    return dataloader.DataLoader(dataset=dataset, batch_size=batch_size,
                                 shuffle=Shuffle, drop_last=True,  pin_memory=pin_memory)

SYNSET_TO_NAME= dict((e[:9], e[10:]) for e in synset.splitlines())
SYNSET_TO_CLASS_ID = dict((e[:9], i) for i, e in enumerate(synset.splitlines()))

CLASS_ID_TO_SYNSET = {v:k for k,v in SYNSET_TO_CLASS_ID.items()}
CLASS_ID_TO_NAME = {i:SYNSET_TO_NAME[CLASS_ID_TO_SYNSET[i]] for i in CLASS_ID_TO_SYNSET}
CLASS_NAME_TO_ID = {v:k for k, v in CLASS_ID_TO_NAME.items()}

#test()






