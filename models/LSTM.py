import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, config: dict):
        super(Model, self).__init__()
        self.config = config

        self.lstm = nn.LSTM(
            input_size=config["lstm_input_size"],
            hidden_size=config["lstm_hidden_size"],
            num_layers=1,
            bias=True,
            batch_first=True,
        )
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(config["lstm_hidden_size"], config["linear_output_size"])
        self.relu = nn.ReLU()
        self.double()

    def forward(self, x: torch.Tensor):
        out, (hidden, cell) = self.lstm(x)
        out = out[:, -1, :]
        out = self.tanh(out)
        out = self.fc(out)
        return out
