import argparse
import os
import torch.nn as nn
from model import GLSE, model_utils,CAKE,STRE
import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class_names = [
        'Seagrass bed', 'Spartina alterniflora', 'Reed', 'Tamarix mixed area',
        'Tidal flat', 'Suaeda salsa', 'Pond', 'Huanghe River', 'Sea', 'Cloud'
    ]

entity_map = {
        "Seagrass bed": 1,
        "Spartina alterniflora": 2,
        "Reed": 3,
        "Tamarix mixed area": 4,
        "Tidal flat": 5,
        "Suaeda salsa": 6,
        "Pond": 7,
        "Huanghe River": 8,
        "Sea": 9,
        "Cloud": 10
    }

class CRLNet(nn.Module):
    def __init__(self,args):
        super(CRLNet, self).__init__()

        self.GLSE = GLSE.GLSE(output_channels=10)
        
        self.CSM = model_utils.CSM()
        self.expand_dim = model_utils.expandDim()
        self.getnode_feats = model_utils.getNode_feats()

        self.STRE = STRE.STRE("./subG/relation_huang2.csv", entity_map, in_feats=128, h_feats=128, excluded_category="Cloud")

        self.Bert = CAKE.KnowledgeGraphEmbedding("/sunhuan/20230408/MyModel/bert-base-uncased", "./subG/attribute.csv", class_names, 128)
        self.CAKE = CAKE.CAKE()

        self.fuse = model_utils.Fuse(num_class = args.num_class)

    def forward(self,input):
        features = self.GLSE(input)
        soft_region = self.CSM(features)
        features = self.expand_dim(features)
        region_representation = self.getnode_feats(features,soft_region)

        space_representation, topo_loss = self.STRE(region_representation)

        triples, _ = self.Bert(input.shape[0])
        attrib_representation = self.CAKE(region_representation,triples)

        reason_representation = [attrib_representation,space_representation]
        output = self.fuse(features,region_representation,reason_representation)

        return output, topo_loss