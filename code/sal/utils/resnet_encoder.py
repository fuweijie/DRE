__all__ = ['ResNetEncoder', 'resnet50encoder']
import torch
import torch.utils.model_zoo as model_zoo
from .CNN_Models import ResNet, Bottleneck
from .pytorch_fixes import adapt_to_image_domain
from torch.autograd import Variable

class ResNetEncoder(ResNet):
    def forward(self, x):
        s0 = x
        x = self.conv1(s0)
        x = self.bn1(x)
        s1 = self.relu(x)
        x = self.maxpool(s1)
        s2 = self.layer1(x)
        s3 = self.layer2(s2)
        s4 = self.layer3(s3)

        s5 = self.layer4(s4)

        x = self.avgpool(s5)
        sX = x.view(x.size(0), -1)
        sC = self.fc(sX)

        return s0, s1, s2, s3, s4, s5, sX, sC


def resnet50encoder(pretrained=True, require_out_grad=False, **kwargs):
    """Constructs a ResNet-50 encoder that returns all the intermediate feature maps.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if require_out_grad==False:
        model = ResNetEncoder(Bottleneck, [3, 4, 6, 3], **kwargs)
    else:
        model = ResNetEncoder_Grad(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth'))
    return model

def get_resnet50encoder_black_box_fn():
    ''' You can try any model from the pytorch model zoo (torchvision.models)
        eg. VGG, inception, mobilenet, alexnet...
    '''
    black_box_model = resnet50encoder(pretrained=True)

    black_box_model.train(False)
    black_box_model = torch.nn.DataParallel(black_box_model)#.cuda()

    def black_box_fn(_images):
        return black_box_model(_images)[-1]
    return black_box_fn