import torch.nn as nn
from torchvision import models
import torch
from torch.nn import functional as F




resnet_dict = {"ResNet18": models.resnet18, "ResNet34": models.resnet34, "ResNet50": models.resnet50,
               "ResNet101": models.resnet101, "ResNet152": models.resnet152}


class ResNet_DHMI(nn.Module):
    def __init__(self, hash_bit, init_bits=2048,res_model="ResNet50"):
        super(ResNet_DHMI, self).__init__()
        model_resnet = resnet_dict[res_model](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                                            self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)
        self.hash_encoder_activate = nn.ReLU(inplace=True)
        self.hash_encoder_layer_L = nn.Linear(model_resnet.fc.in_features, init_bits)
        self.hash_encoder_layer_L.weight.data.normal_(0, 0.01)
        self.hash_encoder_layer_L.bias.data.fill_(0.0)
        self.hash_encoder_layer_L2 = nn.Linear(init_bits, init_bits)

        mid_num = 512
        self.hash_encoder_layer_M = nn.Linear(init_bits, mid_num)
        self.norm_M1 = nn.BatchNorm1d(init_bits, momentum=0.1)
        self.norm_M2 = nn.BatchNorm1d(init_bits, momentum=0.1)
        self.Relu = nn.ReLU(inplace=True)
        self.hash_encoder_layer_S = nn.Linear(mid_num, hash_bit)
        # self.hash_layer = nn.Linear(mid_num, hash_bit)
        self.hash_encoder_layer_S.weight.data.normal_(0, 0.01)
        self.hash_encoder_layer_S.bias.data.fill_(0.0)
        self.W = nn.Parameter(torch.ones([1,2048]))
        self.Sigmoid = nn.Sigmoid()

    def forward(self, input):
        with torch.no_grad():
            x = self.feature_layers(input)
        x = x.view(x.size(0), -1)
        x_L = self.hash_encoder_layer_L(x)
        x_L = x_L.tanh()
        x_M = self.hash_encoder_layer_M(x_L)

        x_M = x_M.tanh()
        x_S = self.hash_encoder_layer_S(x_M)
        x_S = x_S.tanh()
        return x,x_L,x_S


def weights_init(m):
    nn.init.xavier_uniform(m.weight.data)
    nn.init.constant(m.bias.data, 0.00)

class Label_net(nn.Module):
    def __init__(self, label_dim, bit):
        super(Label_net, self).__init__()
        self.module_name = "text_model"
        # 400
        self.cl1 = nn.Linear(2048, label_dim)
        cl3 = nn.Linear(label_dim, 512)
        cl2 = nn.Linear(512, bit)
        nn.Tanh(),
        self.cl_text = nn.Sequential(

            cl3,
            nn.Tanh(),
            cl2,
            nn.Tanh()
        )
    def forward(self, x):
        x_L = self.cl1(x)

        y = self.cl_text(x_L)
        return x_L,y
