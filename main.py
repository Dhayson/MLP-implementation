import numpy as np
import pandas as pd
from src.MLP import MLP, InitializationType
from src.activation_functions import ReLU, Sigmoid, Linear, Tahn, LeakyReLU
from src.loss_functions import MSE, MAE

def main():
    # Carregar o dataset
    data = np.genfromtxt("dataset/bezdekIris.data", delimiter=',', dtype=None, encoding='utf-8')
    features = []
    labels = []
    for i1, i2, i3, i4, label in data:
        features.append([i1, i2, i3, i4])
        labels.append(label)
    features=pd.DataFrame(features)
    labels=pd.DataFrame(labels)
    labels["Iris-setosa"] = labels[0].map({"Iris-setosa": 1, "Iris-versicolor": 0, "Iris-virginica": 0})
    labels["Iris-versicolor"] = labels[0].map({"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 0})
    labels["Iris-virginica"] = labels[0].map({"Iris-setosa": 0, "Iris-versicolor": 0, "Iris-virginica": 1})
    labels = labels.drop(labels=0, axis=1)
    
    mlp = MLP(4, [4,3], 3, [Sigmoid(), Sigmoid(), Sigmoid()], MSE())
    mlp.initialize(InitializationType.gaussian)
    # for i in range(3):
    #     for x in mlp.weight_tensor[i]:
    #         print(*x, sep=" ")
    #     for x in mlp.bias_tensor[i]:
    #         print(x, end=" ")
    #     print()
    #     print()
    # print()
    # print(features, labels)
    loss = 9999
    for i in features.index:
        print(features.loc[i].to_numpy())
        prediction = mlp.predict(features.loc[i].to_numpy())
        print(prediction, labels.loc[i].to_numpy())
        print()
    while True:
        for i in range(200):
            loss = mlp.train(features, labels, sample="Minibatch", n=30)
            if loss < 0.03:
                break
        print(loss)
        if loss < 0.05:
            break
    for i in features.index:
        print(features.loc[i].to_numpy())
        prediction = mlp.predict(features.loc[i].to_numpy())
        print(prediction, labels.loc[i].to_numpy())
        print()
        # print(f"loss: {mlp.loss.loss(prediction, labels.loc[i].to_numpy())}")
        # print(f"loss derivative: {mlp.loss.loss_der(prediction, labels.loc[i].to_numpy())}")
        # loss, output, activate, intermediate = mlp.forward_propagate(features.loc[i].to_numpy(), labels.loc[i].to_numpy())
        # grad_w, grad_b = mlp.backward_propagation(output, labels.loc[i].to_numpy(), activate, intermediate)
        # print(grad_b)
        # print(grad_w)

    

if __name__ == "__main__":
    main()