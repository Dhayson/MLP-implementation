from src.problems import classification_problem, regression_problem
from src_torch.problems import classification_problem_torch, regression_problem_torch
from src.MLP import MLP
from src.activation_functions import *
from src.loss_functions import *
from src.optimization import *
from src.initialization import *
import sys
import torch
from torch import nn
from torch.optim import adagrad
import src_torch.loss_functions

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if sys.argv[1] == "Classification":
        mlp = MLP(4, [4, 4], 3, [ReLU(), Sigmoid(), Sigmoid()], CrossEntropy(), Adagrad(0.5, 600, do_print=(False, 1200)))
        mlp.initialize(GaussianInitialization())
        classification_problem(
            mlp,
            lr = 0.5,
            gradient_type=("Minibatch", 30),
            train_loss_stop = 0.15,
            max_iterations=10000,
            show_each_n_steps=200,
            detail=4
        )
        
    elif sys.argv[1] == "Regression":
        mlp = MLP(43, [12, 12], 3, [ReLU(), ReLU(), Linear()], MSE(), Adagrad(0.5, 400, do_print=(False, 400)))
        mlp.initialize(GaussianInitialization())
        regression_problem(
            # Define os hiperparâmetros da MLP
            # A MLP é capaz de ter um número M de camadas, cada uma com sua própria dimensão e função de ativação
            mlp,
            lr = 0.3,
            gradient_type=("Minibatch", 40),
            subject="Por",
            train_loss_stop=0.01,
            max_iterations=3000,
            set_target=["G1", "G2", "G3"],
            detail=25
        )
        
    elif sys.argv[1] == "ClassificationTorch":
        mlp = nn.Sequential(
            nn.Linear(4, 4),
            nn.ReLU(),
            nn.Linear(4, 4),
            nn.Sigmoid(),
            nn.Linear(4, 3),
            nn.Sigmoid()
        ).to(device)
        def _init_weights(module):
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=sqrt(2/(module.in_features+module.out_features)))
                if module.bias is not None:
                    module.bias.data.zero_()
        mlp.apply(_init_weights)
        optimizer_mlp = adagrad.Adagrad(mlp.parameters(), lr = 0.05)
        classification_problem_torch(
            mlp,
            optimizer_mlp,
            src_torch.loss_functions.CrossEntropy(),
            gradient_type=("Minibatch", 30),
            train_loss_stop=0.0,
            max_iterations=5000,
            show_each_n_steps=200,
            detail=1,
            device=device
        )
        
    elif sys.argv[1] == "RegressionTorch":
        mlp = nn.Sequential(
            nn.Linear(45, 12),
            nn.ReLU(),
            nn.Linear(12, 12),
            nn.ReLU(),
            nn.Linear(12, 1)
        ).to(device)
        def _init_weights(module):
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=sqrt(2/(module.in_features+module.out_features)))
                if module.bias is not None:
                    module.bias.data.zero_()
        mlp.apply(_init_weights)
        optimizer_mlp = adagrad.Adagrad(mlp.parameters(), lr = 0.01)
        regression_problem_torch(
            mlp,
            optimizer_mlp,
            src_torch.loss_functions.MSE(),
            gradient_type=("Minibatch", 40),
            subject="Por",
            train_loss_stop=0.0,
            device=device,
            max_iterations=30000,
            set_target=["G3"],
            detail=25
        )
        

if __name__ == "__main__":
    main()