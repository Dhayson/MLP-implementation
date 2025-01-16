from src.problems import classification_problem, regression_problem
from src.MLP import MLP
from src.activation_functions import *
from src.loss_functions import *
from src.optimization import *
from src.initialization import *
import sys

def main():
    if sys.argv[1] == "Classification":
        mlp = MLP(4, [4, 4], 3, [ReLU(), Sigmoid(), SigmoidBeforeCE()], CrossEntropyAfterSigmoid(), Adagrad(0.5, 600, do_print=(False, 1200)))
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

if __name__ == "__main__":
    main()