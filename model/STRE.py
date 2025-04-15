import torch
import torch.nn as nn
import pandas as pd
import dgl
from dgl.nn import GraphConv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GCN(nn.Module):
    def __init__(self, in_feats=64, h_feats=128):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats, allow_zero_in_degree=True)
        self.conv2 = GraphConv(h_feats, h_feats, allow_zero_in_degree=True)
        self.relu = nn.ReLU()

    def forward(self, graph, features):
        h = self.relu(self.conv1(graph, features))
        h = self.conv2(graph, h)
        return h

class STRE(nn.Module):
    def __init__(self, csv_file_path, entity_map, in_feats=128, h_feats=128, excluded_category=None):
        super(STRE, self).__init__()
        self.gcn_near = GCN(in_feats, h_feats)
        self.gcn_far = GCN(in_feats, h_feats)

        self.alpha = nn.Parameter(torch.tensor(0.5, requires_grad=True))
        self.margin = nn.Parameter(torch.tensor(1.0, requires_grad=True))

        self.u_near, self.v_near, self.u_far, self.v_far = self._load_relation_tensors(csv_file_path, entity_map, excluded_category)

        self.excluded_category = excluded_category
        if excluded_category is not None and excluded_category in entity_map:
            self.excluded_index = entity_map[excluded_category] - 1
        else:
            self.excluded_index = None

    def _load_relation_tensors(self, file_path, entity_map, excluded_category):
        df = pd.read_csv(file_path)
        u_near, v_near, u_far, v_far = [], [], [], []
        excluded_indices = set([entity_map[excluded_category]]) if excluded_category in entity_map else set()

        for _, row in df.iterrows():
            head, relationship, tail = row['head'], row['relation'], row['tail']
            if head in entity_map and tail in entity_map:
                head_index = entity_map[head]
                tail_index = entity_map[tail]
                if head_index in excluded_indices or tail_index in excluded_indices:
                    continue
                if relationship == 'touches':
                    u_near.append(head_index)
                    v_near.append(tail_index)
                elif relationship == 'disjoint':
                    u_far.append(head_index)
                    v_far.append(tail_index)
        return (
            torch.tensor(u_near, dtype=torch.long),
            torch.tensor(v_near, dtype=torch.long),
            torch.tensor(u_far, dtype=torch.long),
            torch.tensor(v_far, dtype=torch.long),
        )

    def create_graph(self, u, v, num_nodes):
        u, v = u - 1, v - 1 
        graph = dgl.graph((u, v), num_nodes=num_nodes)
        graph = dgl.to_bidirected(graph)
        graph = graph.to(device)
        return graph

    def compute_relation_loss(self, node_feats, u, v, relation_type):
        valid_mask = (u < node_feats.size(0)) & (v < node_feats.size(0))
        u, v = u[valid_mask], v[valid_mask]
        if len(u) == 0 or len(v) == 0:
            return torch.tensor(0.0, device=node_feats.device)
        dist = torch.norm(node_feats[u] - node_feats[v], dim=-1)
        if relation_type == "touches":
            return torch.mean(dist**2)
        elif relation_type == "disjoint":
            return torch.mean(torch.relu(self.margin - dist)**2)

    def compute_weighted_loss(self, touches_loss, disjoint_loss):
        alpha = torch.sigmoid(self.alpha)
        weighted_loss = alpha * touches_loss + (1 - alpha) * disjoint_loss
        return weighted_loss

    def forward(self, node_feats):
        n, c, d = node_feats.shape
        if self.excluded_index is not None:
            mask = torch.ones(c, dtype=torch.bool, device=node_feats.device)
            mask[self.excluded_index] = False
            masked_node_feats = node_feats[:, mask, :]
        else:
            masked_node_feats = node_feats
        num_nodes = masked_node_feats.shape[1]
        near_graph_list, far_graph_list = [], []
        near_graph_base = self.create_graph(self.u_near, self.v_near, num_nodes)
        far_graph_base = self.create_graph(self.u_far, self.v_far, num_nodes)
        for i in range(n):
            near_graph = near_graph_base
            near_graph.ndata['features'] = masked_node_feats[i, :, :]
            near_graph_list.append(near_graph)

            far_graph = far_graph_base
            far_graph.ndata['features'] = masked_node_feats[i, :, :]
            far_graph_list.append(far_graph)
        batch_near_g = dgl.batch(near_graph_list)
        batch_far_g = dgl.batch(far_graph_list)
        near_features = self.gcn_near(batch_near_g, batch_near_g.ndata['features'])
        batch_far_g.ndata['features'] = near_features
        far_features = self.gcn_far(batch_far_g, batch_far_g.ndata['features'])
        new_node_feats = torch.stack(
            [far_features[i * num_nodes:(i + 1) * num_nodes] for i in range(n)], dim=0
        )
        flat_node_feats = masked_node_feats.view(-1, d)
        touches_loss = self.compute_relation_loss(flat_node_feats, self.u_near - 1, self.v_near - 1, "touches")
        disjoint_loss = self.compute_relation_loss(flat_node_feats, self.u_far - 1, self.v_far - 1, "disjoint")
        if self.excluded_index is not None:
            final_node_feats = torch.zeros_like(node_feats)
            final_node_feats[:, mask, :] = new_node_feats
            final_node_feats[:, self.excluded_index:self.excluded_index + 1, :] = node_feats[:, self.excluded_index:self.excluded_index + 1, :]
        else:
            final_node_feats = new_node_feats
        topo_loss = self.compute_weighted_loss(touches_loss, disjoint_loss)
        return final_node_feats, topo_loss
