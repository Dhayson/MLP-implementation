import numpy as np
import pandas as pd
from src.MLP import MLP, InitializationType
from src.activation_functions import ReLu, Sigmoid, Linear

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
    
    mlp = MLP(4, [2,2], 3, [Sigmoid, Sigmoid, Linear])
    mlp.initialize(InitializationType.gaussian)
    # for i in range(3):
    #     for x in mlp.weight_tensor[i]:
    #         print(*x, sep=" ")
    #     for x in mlp.bias_tensor[i]:
    #         print(x, end=" ")
    #     print()
    #     print()
    # print()
    
    for i in range(len(features)):
        # print(features.loc[i].array)
        prediction = mlp.predict(features.loc[i].array)
        print(prediction)
        print(f"loss: {mlp.loss(prediction, labels.loc[i].array)}")

    

if __name__ == "__main__":
    main()