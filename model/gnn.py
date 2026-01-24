import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphNeuralNetwork(nn.Module):
    def __init__(self, N_fingerprints, dim, layer_hidden, layer_output, device, feature_dim=9):
        super().__init__()
        self.device = device
        self.dim = dim

        self.embed_fingerprint = nn.Embedding(N_fingerprints, dim)
        self.W_fingerprint = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layer_hidden)])
        self.attn_fc = nn.Linear(dim, 1)
        self.feature_fc = nn.Linear(feature_dim, dim)
        self.fusion_proj = nn.Linear(dim * 2, dim)

        for w in self.W_fingerprint:
            nn.init.xavier_uniform_(w.weight)
            nn.init.zeros_(w.bias)
        nn.init.xavier_uniform_(self.feature_fc.weight)
        nn.init.zeros_(self.feature_fc.bias)
        nn.init.xavier_uniform_(self.fusion_proj.weight)
        nn.init.zeros_(self.fusion_proj.bias)

    def update(self, adjacency, vectors, layer, _dependency_ignored=None):
        h = self.W_fingerprint[layer](vectors)
        if h.dim() == 2:
            attn = F.softmax(h @ h.t(), dim=-1)
            out = attn @ h
        else:
            attn = torch.softmax(torch.bmm(h, h.transpose(1, 2)), dim=-1)  # (B,N,N)
            out  = torch.bmm(attn, h)                                      # (B,N,D)
        return F.relu(out)

    def _ensure_3d(self, x):
        return (x.unsqueeze(0), True) if x.dim() == 2 else (x, False)

    def gnn(self, fingerprints, adjacencies, molecular_sizes, _dependency_ignored=None):
        fps = torch.clamp(fingerprints, 0, self.embed_fingerprint.num_embeddings - 1)
        x = self.embed_fingerprint(fps)
        x3, was_2d = self._ensure_3d(x)
        for l in range(len(self.W_fingerprint)):
            x3 = F.normalize(self.update(adjacencies, x3, l, None), p=2, dim=-1)
        return x3.squeeze(0) if was_2d else x3

    def pool_global(self, node_vectors):
        x3, was_2d = self._ensure_3d(node_vectors)
        w = torch.softmax(self.attn_fc(x3).squeeze(-1), dim=1)
        g = torch.sum(x3 * w.unsqueeze(-1), dim=1)
        return g.squeeze(0) if was_2d else g

    def forward(self, fingerprints, adjacencies, molecular_sizes, dependency_ignored, feature_tensor):
        node_vectors = self.gnn(fingerprints, adjacencies, molecular_sizes, None)

        if feature_tensor is None:
            merged = torch.cat([node_vectors, node_vectors], dim=-1)
            return self.fusion_proj(merged)

        feat = self.feature_fc(feature_tensor)
        node3, was_2d = self._ensure_3d(node_vectors)

        if feat.dim() == 1:
            feat = feat.unsqueeze(0)
        if feat.size(0) != node3.size(0):
            feat = feat.expand(node3.size(0), feat.size(-1))

        feat3 = feat.unsqueeze(1).expand(node3.size(0), node3.size(1), feat.size(-1))
        out = self.fusion_proj(torch.cat([node3, feat3], dim=-1))
        return out.squeeze(0) if was_2d else out
