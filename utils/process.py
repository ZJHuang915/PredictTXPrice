from typing import Callable

import torch
from tqdm import tqdm


def modelTraining(
    model: torch.nn.Module,
    loss_fn: Callable,
    optimizer: torch.optim,
    data_loader: torch.utils.data.DataLoader,
    **kwargs,
):
    if len(data_loader.sampler) == 0:
        return
    device = next(model.parameters()).device
    model.train()
    running_loss = 0.0
    for i, data in enumerate(tqdm(data_loader)):
        # inputs.shape: [batch, len_input, feats]
        # labels.shape: [batch, num_for_predict]
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        # outputs shape: [batch, company, num_for_predict]
        outputs = model(inputs)

        if outputs.shape != labels.shape:
            outputs = outputs.unsqueeze(-1)
            outputs = outputs.float()
            labels = labels.float()

        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    return running_loss / len(data_loader.sampler)


def modelEvaluation(
    model: torch.nn.Module,
    eval_fn: Callable,
    data_loader: torch.utils.data.DataLoader,
    **kwargs,
):
    if len(data_loader.sampler) == 0:
        return

    device = next(model.parameters()).device
    running_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader)):
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            # outputs-shape: [batch, num_for_predict]
            outputs = model(inputs)

            if outputs.shape != labels.shape:
                outputs = outputs.unsqueeze(-1)
                outputs = outputs.float()
                labels = labels.float()

            loss = eval_fn(outputs, labels)
            running_loss += loss.item()
    return running_loss / len(data_loader.sampler)
