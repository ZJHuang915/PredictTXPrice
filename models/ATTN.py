import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, config: dict):
        super(Model, self).__init__()
        self.config = config

        self.Q = nn.Linear(
            in_features=config["input_dim"],
            out_features=config["embed_dim"],
        )
        self.K = nn.Linear(
            in_features=config["input_dim"],
            out_features=config["embed_dim"],
        )
        self.V = nn.Linear(
            in_features=config["input_dim"],
            out_features=config["embed_dim"],
        )

        self.attn = nn.MultiheadAttention(
            embed_dim=config["embed_dim"],
            num_heads=config["num_heads"],
            batch_first=True,
            bias=True,
            add_bias_kv=True,
        )

        self.fc = nn.Linear(
            in_features=config["embed_dim"],
            out_features=config["output_size"],
        )

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.double()

    def forward(self, x: torch.Tensor):
        q, k, v = self.Q(x), self.K(x), self.V(x)
        q, k, v = self.relu(q), self.relu(k), self.relu(v)
        # attn_out-shape: [batch, seq_len, embedd_size]
        attn_out, attn_weight = self.attn(q, k, v)
        # get last slice: [batch, seq_len, embedd_size] -> [batch, embedd_size]
        attn_out = attn_out[:, -1, :]
        attn_out = self.fc(attn_out)
        attn_out = self.tanh(attn_out)
        return attn_out
