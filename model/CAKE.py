import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
import pandas as pd

class KnowledgeGraphEmbedding(nn.Module):
    def __init__(self, model_path, csv_path, class_names, embedding_dim=128):
        super(KnowledgeGraphEmbedding, self).__init__()
        self.model_path = model_path
        self.csv_path = csv_path
        self.class_names = class_names
        self.embedding_dim = embedding_dim

        self.reduce_dim_mlp = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )

        self.df = pd.read_csv(self.csv_path)
        self.filtered_df = self.df[self.df['entity'].isin(self.class_names)]

        self.attention_layer = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=4)
        self.attention_weights = nn.Parameter(torch.rand(len(class_names)))
        self.embedding_cache = {}

    def sentence_to_vector(self, sentence, device):
        if sentence in self.embedding_cache:
            return self.embedding_cache[sentence].to(device)

        tokenizer = BertTokenizer.from_pretrained(self.model_path)
        model = BertModel.from_pretrained(self.model_path).to(device)
        model.eval()

        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        sentence_embedding = outputs.last_hidden_state[:, 0, :].squeeze().to(device)

        self.embedding_cache[sentence] = sentence_embedding
        return sentence_embedding

    def reduce_dimension(self, embedding, device):
        return self.reduce_dim_mlp(embedding.to(device))

    def get_entity_embedding(self, embeddings, device):
        embeddings_tensor = torch.stack(embeddings).to(device)
        num_sentences = embeddings_tensor.size(0)

        attn_output, _ = self.attention_layer(embeddings_tensor.unsqueeze(0), embeddings_tensor.unsqueeze(0),
                                              embeddings_tensor.unsqueeze(0))

        attention_weights = self.attention_weights[:num_sentences].view(-1, 1).to(device)
        weighted_attn_output = attn_output.squeeze(0) * attention_weights

        entity_embedding = torch.mean(weighted_attn_output, dim=0)
        return entity_embedding

    def forward(self, batch_size):
        device = next(self.parameters()).device
        entity_embeddings = []
        for entity in self.class_names:
            entity_rows = self.filtered_df[self.filtered_df['entity'] == entity]
            embeddings = []
            for _, row in entity_rows.iterrows():
                attribute = row['attribute']
                value = row['value']
                sentence = f"{entity}'s {attribute} is {value}."

                embedding = self.sentence_to_vector(sentence, device)

                embedding_128 = self.reduce_dimension(embedding, device)
                embeddings.append(embedding_128)

            entity_embedding = self.get_entity_embedding(embeddings, device)
            entity_embeddings.append(entity_embedding)

        embeddings_tensor = torch.stack(entity_embeddings).to(device)

        batch_embeddings = embeddings_tensor.unsqueeze(0).repeat(batch_size, 1, 1).to(device)

        return batch_embeddings, embeddings_tensor

class Map2Space(nn.Module):
    def __init__(self, in_dim=128, out_dim=128):
        super(Map2Space, self).__init__()
        self.map1 = nn.Linear(in_dim, out_dim)
        self.norm = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()
        self.map2 = nn.Linear(out_dim, out_dim)

    def forward(self, region_representation):
        n, c, d = region_representation.shape
        x = self.map1(region_representation.reshape(-1, d))
        x = self.norm(x)
        x = self.relu(x)
        x = self.map2(x)
        return x.reshape(n, c, -1)

class InteractModule(nn.Module):
    def __init__(self, embedding_dim):
        super(InteractModule, self).__init__()
        self.feature_proj = Map2Space(embedding_dim, embedding_dim)
        self.triple_proj = Map2Space(embedding_dim, embedding_dim)
        
        self.gate_linear = nn.Linear(embedding_dim * 2, embedding_dim)

    def forward(self, features, triples):
        projected_features = features
        projected_triples = triples

        combined = torch.cat([projected_features, projected_triples], dim=-1)
        gate = torch.sigmoid(self.gate_linear(combined))

        optimized_triples = gate * projected_features + (1 - gate) * projected_triples

        return optimized_triples


class CAKE(nn.Module):
    def __init__(self, C=10, D=128, num_heads=4):
        super(CAKE, self).__init__()
        self.num_heads = num_heads
        self.head_dim = D // num_heads
        assert D % num_heads == 0, "Embedding dimension must be divisible by number of heads."

        self.Wq = nn.Parameter(torch.rand(num_heads, D, self.head_dim))
        self.Wk = nn.Parameter(torch.rand(num_heads, D, self.head_dim))
        self.Wv = nn.Parameter(torch.rand(num_heads, D, self.head_dim))

        self.output_linear = nn.Linear(D, D)
        self.norm1 = nn.LayerNorm(D)
        self.norm2 = nn.LayerNorm(D)
        self.interact = InteractModule(D)

    def forward(self, features, triples):
        optimized_triples = self.interact(features, triples)

        features_q = torch.einsum("bcd,hde->bche", optimized_triples, self.Wq)
        features_k = torch.einsum("bcd,hde->bche", features, self.Wk)
        features_v = torch.einsum("bcd,hde->bche", features, self.Wv)

        x = torch.einsum("bche,bche->bhc", features_q, features_k) / (self.head_dim ** 0.5)
        score = F.softmax(x, dim=-1)

        out = torch.einsum("bhc,bche->bche", score, features_v)
        out = out.reshape(out.shape[0], out.shape[1], -1)

        out = self.output_linear(out)
        out = self.norm1(out + optimized_triples)
        out = self.norm2(out + out)

        return out