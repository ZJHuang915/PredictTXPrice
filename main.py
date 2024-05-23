import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from ta import add_all_ta_features
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing_extensions import Annotated

from models import ATTN, LSTM

# from models import ATTN, CNNLSTM, LSTM
from utils import process

# 4 year -> 1 year
config = {
    "shift_day": 1,
    "test_size": 0.2,
    "random_seed": 1,
    "window_size": 10,
    "shuffle_dataset": True,
    "test_start_idx": None,
    "batch_size": 32,
    "learning_rate": 0.001,
    "min_date": "2015-01-01",
    "max_date": "2019-12-31",
    "output_size": 1,
    "input_size": 88,
    "epoch_nums": 201,
    "lstm_hidden_size": 16,
    "attn_embed_dim": 64,
    "attn_num_heads": 1,
    "model": "LSTM",
    # "model": "ATTN",
    # "model": "CNNLSTM",
}


def splitArrayAccordSliceWindow(
    array: Annotated[np.ndarray, "features data shape:[times, feats]"],
    window_size: Annotated[int, "window's size"],
    num_predict: Annotated[int, "number of prediction >= 1"] = 1,
) -> Annotated[Tuple[np.ndarray, np.ndarray], "(X, y)"]:
    l = array.shape[0]
    X, Y = [], []
    for i in range(0, l - window_size + 1 - num_predict):
        x = array[i : (i + window_size)]
        y = array[(i + window_size) : (i + window_size + num_predict)]
        X.append(x)
        Y.append(y)
    return np.stack(X, axis=0), np.stack(Y, axis=0)


class StocksDataset(Dataset):
    def __init__(
        self,
        X: Annotated[torch.Tensor, "shape: [batch, times, feats]"],
        Y: Annotated[torch.Tensor, "shape: [batch, num_predict, feats]"],
        DATE_X: Annotated[torch.Tensor, "shape: [batch, times]"] = None,
        DATE_Y: Annotated[torch.Tensor, "shape: [batch, num_predict]"] = None,
    ) -> None:
        self.X = X
        self.Y = Y
        self.n_samples = X.shape[0]
        self.DATE_X = DATE_X
        self.DATE_Y = DATE_Y

    def __getitem__(self, index: int):
        return self.X[index], self.Y[index]

    def get(self, index: int):
        if self.DATE_X is not None and self.DATE_Y is not None:
            return self.X[index], self.Y[index], self.DATE_X[index], self.DATE_Y[index]

        return self.X[index], self.Y[index]

    def __len__(self):
        return self.n_samples


def getTrainTestDataIndices(
    dataset: Annotated[StocksDataset, "dataset"],
    split_config: Annotated[Dict[str, Any], "dict of config"] = {
        "test_size": 0.2,
        "shuffle_dataset": True,
        "random_seed": 1,
        "test_start_idx": None,
    },
) -> Tuple[List[int], List[int]]:

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor((1.0 - split_config["test_size"]) * dataset_size))
    if "test_start_idx" in split_config and split_config["test_start_idx"] is not None:
        split = split_config["test_start_idx"]

    train_indices, test_indices = indices[:split], indices[split:]
    if split_config["shuffle_dataset"]:
        np.random.seed(split_config["random_seed"])
        np.random.shuffle(train_indices)

    return train_indices, test_indices


def getTrainTestDataSampler(
    dataset: Annotated[StocksDataset, "dataset"],
    split_config: Annotated[Dict[str, Any], "dict of config"] = {
        "test_size": 0.2,
        "shuffle_dataset": True,
        "random_seed": 1,
        "test_start_idx": None,
    },
) -> Tuple[
    Annotated[np.ndarray, "indices of training data"],
    Annotated[np.ndarray, "indices of testing data"],
    Annotated[SubsetRandomSampler, "sampler of training data"],
    Annotated[SubsetRandomSampler, "sampler of testing data"],
]:

    train_indices, test_indices = getTrainTestDataIndices(dataset, split_config)
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    return train_indices, test_indices, train_sampler, test_sampler


if os.path.isfile("./data/台指期2001_2019_1分K/all.csv"):
    df = pd.read_csv("./data/台指期2001_2019_1分K/all.csv")
else:
    file_paths = os.listdir("./data/台指期2001_2019_1分K")
    file_paths = sorted([f for f in file_paths if f[-4:] == ".zip"])

    col_names = ["0", "1", "Date", "Open", "High", "Low", "Close", "Volume"]
    data_list = []
    for file_path in tqdm(file_paths):
        data_list.append(
            pd.read_csv(f"./data/台指期2001_2019_1分K/{file_path}", names=col_names)
        )
    data_df = pd.concat(data_list).reset_index(drop=True)

    df = add_all_ta_features(
        df=data_df,
        open="Open",
        high="High",
        low="Low",
        close="Close",
        volume="Volume",
    )
    df.to_csv("./data/台指期2001_2019_1分K/all.csv", index=False)
df = df[df.Date >= config["min_date"]]
df = df[df.Date <= config["max_date"]]
df.ffill(inplace=True)
df["Date"] = pd.to_datetime(df["Date"])
df["predClose"] = df["Close"].shift(config["shift_day"])
df = df.loc[config["shift_day"] :, :].reset_index(drop=True)
df = df.dropna()


feat_columns = [
    "Open",
    "High",
    "Low",
    "Close",
    "predClose",
    "Volume",
    "volume_adi",
    "volume_obv",
    "volume_cmf",
    "volume_fi",
    "volume_em",
    "volume_sma_em",
    "volume_vpt",
    "volume_vwap",
    "volume_mfi",
    "volume_nvi",
    "volatility_bbm",
    "volatility_bbh",
    "volatility_bbl",
    "volatility_bbw",
    "volatility_bbp",
    "volatility_bbhi",
    "volatility_bbli",
    "volatility_kcc",
    "volatility_kch",
    "volatility_kcl",
    "volatility_kcw",
    # 'volatility_kcp',
    "volatility_kchi",
    "volatility_kcli",
    "volatility_dcl",
    "volatility_dch",
    "volatility_dcm",
    "volatility_dcw",
    "volatility_dcp",
    "volatility_atr",
    "volatility_ui",
    "trend_macd",
    "trend_macd_signal",
    "trend_macd_diff",
    "trend_sma_fast",
    "trend_sma_slow",
    "trend_ema_fast",
    "trend_ema_slow",
    # 'trend_vortex_ind_pos',
    # 'trend_vortex_ind_neg',
    "trend_vortex_ind_diff",
    "trend_trix",
    "trend_mass_index",
    "trend_dpo",
    "trend_kst",
    "trend_kst_sig",
    "trend_kst_diff",
    "trend_ichimoku_conv",
    "trend_ichimoku_base",
    "trend_ichimoku_a",
    "trend_ichimoku_b",
    "trend_stc",
    "trend_adx",
    "trend_adx_pos",
    "trend_adx_neg",
    "trend_cci",
    "trend_visual_ichimoku_a",
    "trend_visual_ichimoku_b",
    "trend_aroon_up",
    "trend_aroon_down",
    "trend_aroon_ind",
    "trend_psar_up",
    "trend_psar_down",
    "trend_psar_up_indicator",
    "trend_psar_down_indicator",
    "momentum_rsi",
    "momentum_stoch_rsi",
    "momentum_stoch_rsi_k",
    "momentum_stoch_rsi_d",
    "momentum_tsi",
    "momentum_uo",
    "momentum_stoch",
    "momentum_stoch_signal",
    "momentum_wr",
    "momentum_ao",
    "momentum_roc",
    "momentum_ppo",
    "momentum_ppo_signal",
    "momentum_ppo_hist",
    "momentum_pvo",
    "momentum_pvo_signal",
    "momentum_pvo_hist",
    "momentum_kama",
    "others_dr",
    "others_dlr",
    "others_cr",
]

scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

train_set_point = int(len(df) * (1 - config["test_size"]))
scaler.fit(df.loc[:train_set_point, feat_columns])  # 對所有特徵
y_scaler.fit(df.loc[:train_set_point, ["Close"]])  # 對收盤價
scaled_arr = scaler.transform(df.loc[:, feat_columns])

X, Y = splitArrayAccordSliceWindow(array=scaled_arr, window_size=config["window_size"])
Date_X, Date_Y = splitArrayAccordSliceWindow(
    array=df["Date"].to_numpy(), window_size=config["window_size"]
)

response_index = feat_columns.index("predClose")
print(f"response_index: {response_index}")

feat_indices = list(range(response_index)) + list(
    range(response_index + 1, len(feat_columns))
)
X = X[:, :, feat_indices]
Y = Y[:, :, response_index]
dataset = StocksDataset(X=X, Y=Y, DATE_X=Date_X, DATE_Y=Date_Y)

dataset_utils = getTrainTestDataSampler(dataset=dataset, split_config=config)
train_indices, test_indices, train_sampler, test_sampler = dataset_utils

train_loader = DataLoader(
    dataset, batch_size=config["batch_size"], sampler=train_sampler
)
test_loader = DataLoader(dataset, batch_size=config["batch_size"], sampler=test_sampler)


if config["model"] == "ATTN":
    model = ATTN.Model(
        config={
            "input_dim": config["input_size"],
            "embed_dim": config["attn_embed_dim"],
            "num_heads": config["attn_num_heads"],
            "output_size": config["output_size"],
        }
    )
elif config["model"] == "LSTM":
    model = LSTM.Model(
        config={
            "lstm_input_size": config["input_size"],
            "lstm_hidden_size": config["lstm_hidden_size"],
            "linear_output_size": config["output_size"],
        }
    )
# elif config["model"] == "CNNLSTM":
#     model = CNNLSTM.Model(feat_size=88, emb_dim=64, seq_length=10, heads=8, dropout=0.5)

print(model)

runs_dir_path = "./runs"
model_params_dir_path = "./model_params"
exp_name = "_".join(
    [
        "TX00",
        config["model"],
        f"min_date-{config['min_date']}",
        f"max_date-{config['max_date']}",
        f"test_size-{config['test_size']}",
        f"batch_size-{config['batch_size']}",
    ]
)
writer = SummaryWriter(f"{runs_dir_path}/{exp_name}")
folder_path = f"{model_params_dir_path}/{exp_name}"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print(f"Folder '{folder_path}' created.")
else:
    print(f"Folder '{folder_path}' already exists.")


def main():
    loss_fn = torch.nn.MSELoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    for epoch in tqdm(range(config["epoch_nums"])):
        train_loss = process.modelTraining(
            model=model, loss_fn=loss_fn, optimizer=optimizer, data_loader=train_loader
        )
        train_mse = process.modelEvaluation(
            model=model, eval_fn=loss_fn, data_loader=train_loader
        )
        test_mse = process.modelEvaluation(
            model=model, eval_fn=loss_fn, data_loader=test_loader
        )
        print(f"[EPOCH - {epoch}]", end=" ")
        print(f"Prev train loss: {train_loss:.08f};", end=" ")
        print(f"Prev train mse: {train_mse:.08f};", end=" ")
        print(f"Prev test mse: {test_mse:.08f};", end=" ")
        writer.add_scalar("LOSS/train", train_loss, epoch)
        writer.add_scalar("MSE/train", train_mse, epoch)
        writer.add_scalar("MSE/test", test_mse, epoch)
        if epoch % 20 == 0 and epoch != 0:
            save_path = f"{folder_path}/model_{epoch}.pth"
            torch.save(model, save_path)


if __name__ == "__main__":
    main()
