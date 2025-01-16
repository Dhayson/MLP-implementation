import numpy as np
import pandas as pd
from src.MLP import MLP
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo

def load_iris_dataset(normalized = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Carregar o dataset
    iris = fetch_ucirepo(id=53) 
    features = iris.data.features
    labels: pd.DataFrame = iris.data.targets
    pd.options.mode.copy_on_write = True
    labels["Iris-setosa"] = labels['class'].map({"Iris-setosa": 1.0, "Iris-versicolor": 0.0, "Iris-virginica": 0.0})
    labels["Iris-versicolor"] = labels['class'].map({"Iris-setosa": 0.0, "Iris-versicolor": 1.0, "Iris-virginica": 0.0})
    labels["Iris-virginica"] = labels['class'].map({"Iris-setosa": 0.0, "Iris-versicolor": 0.0, "Iris-virginica": 1.0})
    labels = labels.drop(labels='class', axis=1)

    return features, labels

def load_student_dataset(subject: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Carregar o dataset
    if subject == "Mat":
        data = pd.read_csv("dataset_students/student-mat.csv", sep=";")
    elif subject == "Por":
        # O do ucirepo é apenas português
        student_performance = fetch_ucirepo(id=320)
        data: pd.DataFrame = student_performance.data.original
        
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

def classification_problem(
    mlp: MLP, 
    lr: float, 
    gradient_type: tuple[str, int],
    train_loss_stop: float,
    max_iterations = 9999999999,
    show_each_n_steps = 144,
    detail = 12
):
    # Carregar dataset
    features, labels = load_iris_dataset()
    
    # Normalização
    normalized = True
    if normalized:
        features=(features-features.min())/(features.max()-features.min())
        
    # Dividir entre treino e validação
    features_train = features.sample(frac=0.6)
    labels_train = labels.loc[features_train.index]
    features_val = features.drop(features_train.index)
    labels_val = labels.loc[features_val.index]
    
    train_losses_t = []
    train_accs_t = []
    val_losses_t = []
    val_accs_t = []
    while True:
        train_loss, train_acc = mlp.eval(features_train, labels_train, kind="Classification")
        train_losses_t.append((mlp.t, train_loss))
        train_accs_t.append((mlp.t, train_acc))
        
        val_loss, val_acc = mlp.eval(features_val, labels_val, kind="Classification")
        val_losses_t.append((mlp.t, val_loss))
        val_accs_t.append((mlp.t, val_acc))
        
        
        if show_each_n_steps != -1 and mlp.t % show_each_n_steps == 0:
            print(f"t = {mlp.t}")
            print(f"train loss: {train_loss} train_acc: {train_acc}")
            print(f"val loss: {train_loss} val_acc: {val_acc}")
        
        if train_loss < train_loss_stop or mlp.t >= max_iterations:
            break
        
        for _i in range(detail):
            train_loss = mlp.train(features_train, labels_train, sample=gradient_type[0], learning_rate=lr, n=gradient_type[1])
            
    train_loss, train_accuracy = mlp.eval(features_train, labels_train, kind="Classification")
    print(f"train loss: {train_loss} train accuracy: {train_accuracy}")
    val_loss, val_accuracy = mlp.eval(features_val, labels_val, kind="Classification")
    print(f"val loss: {val_loss} val accuracy: {val_accuracy}")
    
    f1 = plt.figure(1)
    plt.yscale('log')
    plt.plot([a for a,b in train_losses_t], [b for a,b in train_losses_t], label='train loss', color='blue')
    plt.plot([a for a,b in val_losses_t], [b for a,b in val_losses_t], label='val loss', color='purple')
    plt.xlabel('Iterações')
    plt.ylabel('Loss: treino azul, validação roxo')
    
    f2 = plt.figure(2)
    plt.plot([a for a,b in train_accs_t], [b for a,b in train_accs_t], label='train loss', color='blue')
    plt.plot([a for a,b in val_accs_t], [b for a,b in val_accs_t], label='val loss', color='purple')
    plt.xlabel('Iterações')
    plt.ylabel('Acurácia: treino azul, validação roxo')
    
    plt.show()
    
    if train_accuracy == 1.0 and val_accuracy == 1.0 and show_each_n_steps != -1:
        print()
        print(mlp.weight_tensor)
        print(mlp.bias_tensor)

    
def regression_problem(
    mlp: MLP, 
    lr: float, 
    gradient_type: tuple[str, int], 
    subject: str,
    train_loss_stop: float,
    max_iterations = 9999999999,
    set_target = ["G3"],
    show_each_n_steps = 200,
    detail = 25
):
    # Carregar dataset
    dataset = load_student_dataset(subject)
    
    # Normalização
    tmax = dataset[set_target].max()
    tmin  = dataset[set_target].min()
    normalized = True
    if normalized:
        dataset=(dataset-dataset.min())/(dataset.max()-dataset.min())
    
    # Divisão entre target e features
    target = dataset[set_target]
    features = dataset.drop(set_target, axis=1)
    
    # Dividir entre treino e validação
    features_train = features.sample(frac=0.6)
    target_train = target.loc[features_train.index]
    features_val = features.drop(features_train.index)
    target_val = target.loc[features_val.index]
    
    train_losses_t = []
    train_rmse_t = []
    train_mae_t = []
    val_losses_t = []
    val_rmse_t = []
    val_mae_t = []
    while True:
        train_loss, train_rmse, train_mae = mlp.eval(
            features_train, target_train, kind="Regression", denormalize=True, tmax=tmax, tmin=tmin)
        train_losses_t.append((mlp.t, train_loss))
        train_rmse_t.append((mlp.t, train_rmse))
        train_mae_t.append((mlp.t, train_mae))
        
        val_loss, val_rmse, val_mae = mlp.eval(
            features_val, target_val, kind="Regression", denormalize=True, tmax=tmax, tmin=tmin)
    
        val_loss, _ = mlp.eval(features_val, target_val)
        val_losses_t.append((mlp.t, val_loss))
        val_rmse_t.append((mlp.t, val_rmse))
        val_mae_t.append((mlp.t, val_mae))
        
        if show_each_n_steps != -1 and mlp.t % show_each_n_steps == 0:
            print(f"t = {mlp.t}")
            print(f"train loss: {train_loss} train rmse: {train_rmse} train mae: {train_mae}")
            print(f"val loss: {val_loss} val rmse: {val_rmse} val mae: {val_mae}")
            print()
            
        if train_loss < train_loss_stop or mlp.t >= max_iterations:
            break
        
        for _i in range(detail):
            mlp.train(features_train, target_train, sample=gradient_type[0], learning_rate=lr, n=gradient_type[1])
            
            
    train_loss, train_rmse, train_mae = mlp.eval(features_train, target_train, kind="Regression", denormalize=True, tmax=tmax, tmin=tmin)
    print(f"train loss: {train_loss} train rmse: {train_rmse} train mae: {train_mae}")
    val_loss, val_rmse, val_mae = mlp.eval(features_val, target_val, kind="Regression", denormalize=True, tmax=tmax, tmin=tmin)
    print(f"val loss: {val_loss} val rmse: {val_rmse} val mae: {val_mae}")
    

    f1 = plt.figure(1)
    plt.yscale('log')
    plt.plot([a for a,b in train_rmse_t], [b for a,b in train_rmse_t], label='train rmse', color='blue')
    plt.plot([a for a,b in val_rmse_t], [b for a,b in val_rmse_t], label='val rmse', color='purple')
    plt.xlabel('Iterações')
    plt.ylabel('RMSE: treino azul, validação roxo')
    
    f2 = plt.figure(2)
    plt.yscale('log')
    if len(set_target) == 1:
        plt.plot([a for a,b in train_mae_t], [b for a,b in train_mae_t], label='train mae', color='blue')
        plt.plot([a for a,b in val_mae_t], [b for a,b in val_mae_t], label='val mae', color='purple')
        plt.xlabel('Iterações')
        plt.ylabel('MAE: treino azul, validação roxo')
    elif len(set_target) == 2:
        plt.plot([a for a,b in train_mae_t], [c for c,d in [b for a,b in train_mae_t]], label='train mae', color='blue')
        plt.plot([a for a,b in train_mae_t], [d for c,d in [b for a,b in train_mae_t]], label='train mae', color='green')
        plt.plot([a for a,b in val_mae_t], [c for c,d in [b for a,b in val_mae_t]], label='val mae', color='purple')
        plt.plot([a for a,b in val_mae_t], [d for c,d in [b for a,b in val_mae_t]], label='val mae', color='red')
        plt.xlabel('Iterações')
        plt.ylabel('MAE: treino azul(G2) e verde(G3)\nvalidação roxo(G2) e vermelho(G3)')
    elif len(set_target) == 3:
        plt.plot([a for a,b in train_mae_t], [c for c,d,e in [b for a,b in train_mae_t]], label='train mae', color='blue')
        plt.plot([a for a,b in train_mae_t], [d for c,d,e in [b for a,b in train_mae_t]], label='train mae', color='green')
        plt.plot([a for a,b in train_mae_t], [e for c,d,e in [b for a,b in train_mae_t]], label='train mae', color='teal')
        plt.plot([a for a,b in val_mae_t], [c for c,d,e in [b for a,b in val_mae_t]], label='val mae', color='purple')
        plt.plot([a for a,b in val_mae_t], [d for c,d,e in [b for a,b in val_mae_t]], label='val mae', color='red')
        plt.plot([a for a,b in val_mae_t], [e for c,d,e in [b for a,b in val_mae_t]], label='val mae', color='magenta')
        plt.xlabel('Iterações')
        plt.ylabel('MAE: treino azul(G1), verde(G2) e azul-petróleo(G3)\nvalidação roxo(G1), vermelho(G2) e magenta(G3)')
    plt.show()
    
    
