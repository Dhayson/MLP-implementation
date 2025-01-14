import numpy as np
import pandas as pd
from src.MLP import MLP, InitializationType
from src.activation_functions import ReLU, Sigmoid, SigmoidBeforeCE, Linear
from src.loss_functions import MSE, CrossEntropy, CrossEntropyAfterSigmoid
from src.optimization import Adagrad
import matplotlib.pyplot as plt

def load_iris_dataset(normalized = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Carregar o dataset
    data = np.genfromtxt("dataset_iris/bezdekIris.data", delimiter=',', dtype=None, encoding='utf-8')
    features = []
    labels = []
    for i1, i2, i3, i4, label in data:
        features.append([i1, i2, i3, i4])
        labels.append(label)
    features=pd.DataFrame(features)
    labels=pd.DataFrame(labels)
    labels["Iris-setosa"] = labels[0].map({"Iris-setosa": 1.0, "Iris-versicolor": 0.0, "Iris-virginica": 0.0})
    labels["Iris-versicolor"] = labels[0].map({"Iris-setosa": 0.0, "Iris-versicolor": 1.0, "Iris-virginica": 0.0})
    labels["Iris-virginica"] = labels[0].map({"Iris-setosa": 0.0, "Iris-versicolor": 0.0, "Iris-virginica": 1.0})
    labels = labels.drop(labels=0, axis=1)

    return features, labels

def load_student_dataset() -> pd.DataFrame:
    # Carregar o dataset
    data = pd.read_csv("dataset_students/student-mat.csv", sep=";")
    data["school"] = data["school"].map({"GP":1, "MS":0})
    data["sex"] = data["sex"].map({"F":1, "M":0})
    data["address"] = data["address"].map({"U":1, "R":0})
    data["famsize"] = data["famsize"].map({"GT3":1, "LE3":0})
    data["Pstatus"] = data["Pstatus"].map({"A":1, "T":0})
    yes_no_columns = ["schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet", "romantic"]
    def yes_no_to_int(x):
        if x == "yes":
            return 1
        else:
            return 0
    data[yes_no_columns] = data[yes_no_columns].applymap(yes_no_to_int)
    def check_for(x1, x2):
        if x1 == x2:
            return 1
        else:
            return 0
        
    data["Mjob_is_other"] = data["Mjob"].map(lambda x: check_for(x, "other"))
    data["Mjob_is_services"] = data["Mjob"].map(lambda x: check_for(x, "services"))
    data["Mjob_is_teacher"] = data["Mjob"].map(lambda x: check_for(x, "teacher"))
    data["Mjob_is_at_home"] = data["Mjob"].map(lambda x: check_for(x, "at_home"))
    data["Mjob_is_health"] = data["Mjob"].map(lambda x: check_for(x, "health"))
    
    data["Fjob_is_other"] = data["Fjob"].map(lambda x: check_for(x, "other"))
    data["Fjob_is_services"] = data["Fjob"].map(lambda x: check_for(x, "services"))
    data["Fjob_is_teacher"] = data["Fjob"].map(lambda x: check_for(x, "teacher"))
    data["Fjob_is_at_home"] = data["Fjob"].map(lambda x: check_for(x, "at_home"))
    data["Fjob_is_health"] = data["Fjob"].map(lambda x: check_for(x, "health"))
    
    data["reason_is_course"] = data["reason"].map(lambda x: check_for(x, "course"))
    data["reason_is_home"] = data["reason"].map(lambda x: check_for(x, "home"))
    data["reason_is_reputation"] = data["reason"].map(lambda x: check_for(x, "reputation"))
    data["reason_is_other"] = data["reason"].map(lambda x: check_for(x, "other"))
    
    data["guardian_is_mother"] = data["guardian"].map(lambda x: check_for(x, "mother"))
    data["guardian_is_father"] = data["guardian"].map(lambda x: check_for(x, "father"))
    data["guardian_is_other"] = data["guardian"].map(lambda x: check_for(x, "other"))

    data = data.drop(labels=["Mjob", "Fjob", "reason", "guardian"], axis=1)
    
    return data

def classification_problem():
    features, labels = load_iris_dataset()
    normalized = True
    
    if normalized:
        features=(features-features.min())/(features.max()-features.min())
        
    # Dividir entre treino e validação
    features_train = features.sample(frac=0.6)
    labels_train = labels.loc[features_train.index]
    features_val = features.drop(features_train.index)
    labels_val = labels.loc[features_val.index]
    
    # Define os hiperparâmetros da MLP
    # A MLP é capaz de ter um número M de camadas, cada uma com sua própria dimensão e função de ativação
    
    mlp = MLP(4, [4, 4], 3, [ReLU(), Sigmoid(), SigmoidBeforeCE()], CrossEntropyAfterSigmoid(), Adagrad(0.5, 600, do_print=(False, 1200)))
    mlp.initialize(InitializationType.gaussian)
    loss = 9999
        
    lr = 0.6
    n = 40
    while True:
        loss, train_acc = mlp.eval(features_train, labels_train, kind="Classification")
        print(loss)
        print(mlp.t)
        # if loss < 0.0025:
        #     break
        _, val_acc = mlp.eval(features_val, labels_val, kind="Classification")
        
        if train_acc == val_acc and val_acc == 1.0:
            print("SUCCESS")
            print(mlp.weight_tensor)
            print(mlp.bias_tensor)
            print()
            break
        
        for _i in range(300):
            loss = mlp.train(features_train, labels_train, sample="Minibatch", learning_rate=lr, n=n)
    train_loss, train_accuracy = mlp.eval(features_train, labels_train, kind="Classification")
    print(f"train loss: {train_loss} train accuracy: {train_accuracy}")
    val_loss, val_accuracy = mlp.eval(features_val, labels_val, kind="Classification")
    print(f"val loss: {val_loss} val accuracy: {val_accuracy}")
    if train_accuracy == val_accuracy and val_accuracy == 1.0:
        print()
        print(mlp.weight_tensor)
        print(mlp.bias_tensor)

    
def regression_problem():
    dataset = load_student_dataset()
    set_target = ["G3"]
    
    
    tmax = dataset[set_target].max()
    tmin  = dataset[set_target].min()
    normalized = True
    if normalized:
        dataset=(dataset-dataset.min())/(dataset.max()-dataset.min())
        
    
    target = dataset[set_target]
    features = dataset.drop(set_target, axis=1)
    
    # Dividir entre treino e validação
    features_train = features.sample(frac=0.6)
    target_train = target.loc[features_train.index]
    features_val = features.drop(features_train.index)
    target_val = target.loc[features_val.index]
    
    # Define os hiperparâmetros da MLP
    # A MLP é capaz de ter um número M de camadas, cada uma com sua própria dimensão e função de ativação
    
    mlp = MLP(45, [12, 12], 1, [ReLU(), ReLU(), Linear()], MSE(), Adagrad(0.5, 400, do_print=(False, 400)))
    mlp.initialize(InitializationType.gaussian)
        
    lr = 0.3
    batch_size = 40
    
    train_losses = []
    val_losses = []
    while True:
        train_loss, train_rmse, train_mae = mlp.eval(
            features_train, target_train, kind="Regression", denormalize=True, tmax=tmax, tmin=tmin)
        train_losses.append((mlp.t, train_loss))
        
        val_loss, val_rmse, val_mae = mlp.eval(
            features_val, target_val, kind="Regression", denormalize=True, tmax=tmax, tmin=tmin)
    
        val_loss, _ = mlp.eval(features_val, target_val)
        val_losses.append((mlp.t, val_loss))
        if mlp.t % 200 == 0:
            print(mlp.t)
            print(f"train loss: {train_loss} train rmse: {train_rmse} train mae: {train_mae}")
            print(f"val loss: {val_loss} val rmse: {val_rmse} val mae: {val_mae}")
            print()
        if train_loss < 0.001:
            break
        for _i in range(25):
            mlp.train(features_train, target_train, sample="Minibatch", learning_rate=lr, n=batch_size)
            
            
    train_loss, train_rmse, train_mae = mlp.eval(features_train, target_train, kind="Regression", denormalize=True, tmax=tmax, tmin=tmin)
    print(f"train loss: {train_loss} train rmse: {train_rmse} train mae: {train_mae}")
    val_loss, val_rmse, val_mae = mlp.eval(features_val, target_val, kind="Regression", denormalize=True, tmax=tmax, tmin=tmin)
    print(f"val loss: {val_loss} val rmse: {val_rmse} val mae: {val_mae}")
    
    plt.plot([a for a,b in train_losses], [b for a,b in train_losses], label='train loss', color='blue')
    plt.plot([a for a,b in val_losses], [b for a,b in val_losses], label='val loss', color='purple')
    plt.yscale('log')
    plt.xlabel('Iterações')
    plt.ylabel('Loss: treino azul, validação roxo')
    plt.show()
    
    
