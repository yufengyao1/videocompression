import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import vgg19

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg_model = vgg19(pretrained=False).features[:].to(device)
vgg_model.eval()
for param in vgg_model.parameters():
    param.requires_grad = False  
    
    
class LossNetwork(torch.nn.Module):
    def __init__(self, vgg_model):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg_model
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '13': "relu3_2",
            '22': "relu4_2",
            '31': "relu5_2"
        }
        # self.weight = [1/2.6,1/4.8,1/3.7,1/5.6,10/1.5]
        self.weight = [1.0, 1.0, 1.0, 1.0, 1.0]

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def forward(self, output, gt):
        loss = []
        output_features = self.output_features(output)
        gt_features = self.output_features(gt)
        for iter, (dehaze_feature, gt_feature, loss_weight) in enumerate(zip(output_features, gt_features, self.weight)):
            loss.append(F.mse_loss(dehaze_feature, gt_feature)*loss_weight)
        return sum(loss)


class C3DLoss(nn.Module):
    def __init__(self):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()
        self.l1loss=nn.L1Loss()
        self.criterion_feature=LossNetwork(vgg_model)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1).to(device)

    def forward(self, label, out):
        l1_loss = self.l1loss(label, out)

        label = label[0].permute(1, 0, 2, 3)
        out = out[0].permute(1, 0, 2, 3)

        label = (label - self.mean) / self.std
        out = (out - self.mean) / self.std

        loss_feature = self.criterion_feature(label, out)
        return l1_loss+loss_feature
