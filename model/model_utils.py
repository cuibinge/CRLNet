import torch
import torch.nn as nn
import torch.nn.functional as F


class CSM(nn.Module):
    def __init__(self, num_class=10, in_chs=32, mid_chs=32):
        super(CSM, self).__init__()
        self.conv1 = nn.Conv2d(10, mid_chs, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_chs)
        self.conv2_1 = nn.Conv2d(mid_chs, mid_chs, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv2_2 = nn.Conv2d(mid_chs, mid_chs, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU(inplace=True)
        self.conv_out = nn.Conv2d(mid_chs, num_class, kernel_size=1, stride=1, padding=0)

    def forward(self, features):
        x = self.conv1(features)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.conv2_1(x)
        x2 = self.conv2_2(x)
        x = x1 + x2

        x = self.conv_out(x)
        return x


class expandDim(nn.Module):
    def __init__(self):
        super(expandDim, self).__init__()

        self.expand_dim = nn.Sequential(
            nn.Conv2d(10, 128,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
    def forward(self,features):

        return self.expand_dim(features)


class getNode_feats(nn.Module):
    def __init__(self, scale=1):
        super(getNode_feats, self).__init__()
        self.scale = scale
        self.norm = nn.LayerNorm(128)

    def forward(self, features, soft_regions):
        batch_size, c, h, w = soft_regions.size()
        soft_regions = soft_regions.view(batch_size, c, -1) 
        features = features.view(batch_size, features.size(1), -1) 
        features = features.permute(0, 2, 1)

        soft_regions = F.softmax(self.scale * soft_regions, dim=-1)
        region_representation = torch.matmul(soft_regions, features)

        region_representation = self.norm(region_representation)
        return region_representation
    
class Fuse(nn.Module):
    def __init__(self,num_class = 7):
        super(Fuse, self).__init__()
        self.trans_dim = nn.Linear(128*2,128)
        self.classifier = nn.Sequential(
            nn.Conv2d( num_class, num_class, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_class),
            nn.ReLU(),
            nn.Conv2d(num_class,num_class,kernel_size=1,stride=1,padding=0)
        )
        # self.classifier = classfier_add(in_ch=num_class,mid_ch=num_class)
        self.alpha = nn.Parameter(torch.tensor(0.5))
    def forward(self,deep_feat,region_feat,reasoning_feats):
        n,d,h,w = deep_feat.shape
        # print(deep_feat.shape)
        c = region_feat.size(1)
        reasoning_feat = torch.cat((reasoning_feats[0],reasoning_feats[1]),dim=2)
        reasoning_feat = self.trans_dim(reasoning_feat)
        reasoning_feat = self.alpha * region_feat + (1 - self.alpha) * reasoning_feat
        output1 = torch.matmul(reasoning_feat, deep_feat.view(n, d, -1)).view(n, c, h, w)

        output = self.classifier(output1)
        
        return output