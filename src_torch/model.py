import numpy as np
import pandas as pd
from math import sqrt
from torch import nn
from src_torch.loss_functions import LossFunction
import torch

def train_model(
    mlp: nn.Module, 
    loss_f: LossFunction,
    optimizer: torch.optim.Adagrad, 
    dataset: pd.DataFrame, 
    expected: pd.DataFrame, 
    sample = "Batch", 
    n=-1,
    device = "cpu"
):
    assert len(dataset) == len(expected)
    if sample == "Batch":
        n = len(dataset)
    elif sample == "Stochastic":
        n = 1
    elif sample == "Minibatch":
        n = n
    assert n > 0
    
    # Coleta uma amostra de n elementos aleatórios
    dataset_sample = dataset.sample(n=n)
    dataset_tensor = torch.tensor(dataset_sample.values, device=device).float()
    expected_sample= expected.loc[dataset_sample.index]
    expected_tensor = torch.tensor(expected_sample.values, device=device).float()
    
    dataset_sample = dataset_sample.reset_index(drop=True)
    expected_sample = expected_sample.reset_index(drop=True)
    
    optimizer.zero_grad()
    
    # Calcula a predição da amostra
    outputs = mlp(dataset_tensor)
    
    total_loss = torch.zeros(1, device=device)
    for i in expected_sample.index:
        loss = loss_f.loss(outputs[i], expected_tensor[i], device)
        total_loss = total_loss + loss
        
    total_loss /= torch.tensor(n, device=device)
    
    # Aplica a otimização
    total_loss.backward()
    optimizer.step()
    
    return total_loss


def eval_model(
    mlp: nn.Module, 
    loss_f: LossFunction, 
    dataset: pd.DataFrame, 
    expected: pd.DataFrame, 
    kind="None", 
    denormalize = False, 
    tmax = None, 
    tmin = None, 
    do_print = False, 
    device = "cpu"
):
    """Avalia o modelo em um determinado conjunto

    Args:
        dataset (pd.DataFrame): Features do conjunto a ser avaliado
        expected (pd.DataFrame): Valores esperados
        kind (str, optional): Regression ou Classification. Defaults to "None".
        denormalize (bool, optional): Inverte a normalização. Defaults to False.
        tmax (_type_, optional): Valores máximos antes da normalização. Defaults to None.
        tmin (_type_, optional): Valores mínimos antes da normalização. Defaults to None.
        do_print (bool, optional): Se o modelo realiza prints de debug. Defaults to False.

    Returns:
        Para classificação, retorna loss, accuracy
        
        Para regressão, retorna loss, rmse, e mae
    """
    assert len(dataset) == len(expected)
    n = len(dataset)
    
    dataset = dataset.reset_index(drop=True)
    expected = expected.reset_index(drop=True)
    
    dataset_tensor = torch.tensor(dataset.values, device=device).float()
    expected_tensor = torch.tensor(expected.values, device=device).float()
    
    mlp.eval()
    torch.no_grad()
    
    train_loss = torch.zeros(1, device=device)
    
    classification_pred = []
    classification_exp = []
    
    regression_pred = []
    regression_exp = []

    outputs = mlp(dataset_tensor)
    for i in dataset.index:
        if kind == "Classification":
            classification_pred.append((np.argmax(outputs[i].cpu().detach()), i))
            classification_exp.append(np.argmax(expected.loc[i].to_numpy()))
        if kind == "Regression":
            if denormalize:
                output_i = outputs[i].cpu().detach().numpy()
                regression_pred.append((output_i*(tmax-tmin) + tmin, i))
                regression_exp.append(expected.loc[i].to_numpy()*(tmax-tmin) + tmin)
            else:
                regression_pred.append((outputs[i], i))
                regression_exp.append(expected.loc[i].to_numpy())
        train_loss = train_loss + loss_f.loss(outputs[i], expected_tensor[i], device)
    
    n_wrong = 0
    if kind == "Classification":
        for i in range(len(classification_pred)):
            if classification_pred[i][0] != classification_exp[i]:
                n_wrong += 1
                if do_print:
                    print(f"missed: {classification_pred[i]} expected {classification_exp[i]}")       
            
    rmse = 0
    mae = 0       
    if kind == "Regression":
        for i in range(len(regression_pred)):
            if do_print:
                print(f"{regression_pred[i]}")
                print("expected")
                print(regression_exp[i])
                print()
            l = regression_pred[i][0] - regression_exp[i]
            rmse += l*l
            mae += abs(l)
            
    
    train_loss /= torch.tensor(n, device=device)
    
    mlp.train()
    torch.enable_grad()
    if kind == "Classification":
        accuracy = (n-n_wrong)/n
        return train_loss, accuracy
    elif kind == "Regression":
        rmse = sqrt(rmse.sum()/n)
        mae = mae/n
        return train_loss, rmse, mae
    else:
        return train_loss, None
