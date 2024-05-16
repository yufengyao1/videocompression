import torch
from torch import nn
import torchvision.models as models
from torchvision.models import vgg16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class C3DLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg16 = vgg16(pretrained=True).features[:-3]
        self.vgg16.eval()
        self.criterion = nn.MSELoss()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)

    def extractfeatures(self, input):
        return self.vgg16(input)

    def forward(self, label, out, model):
        l1_loss = self.criterion(label, out)

        label = label[0].permute(1, 0, 2, 3)
        out = out[0].permute(1, 0, 2, 3)

        label = (label - self.mean) / self.std
        out = (out - self.mean) / self.std

        label = self.vgg16(label)
        out = self.vgg16(out)

        l2_loss = self.criterion(label,out)
        return l1_loss+l2_loss
