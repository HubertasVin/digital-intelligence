import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 1. DUOMENU PARUOSIMAS
# 1.1 Nuskaitomi duomenys
def prepare_data(file_path):
    prefix_line = "id,clump_thickness,uniform_cell_size,uniform_cell_shape,margin_adhesion,single_cell_size,bare_nuclei,bland_chromatin,normal_nucleoli,mitoses,class"
    with open(file_path, "r+") as source:
        content = source.read()
        with open("temp.data", "w") as out:
            out.write(prefix_line.rstrip("\r\n") + "\n" + content)
    data = pd.read_csv("temp.data")
    os.remove("temp.data")
    # Pasaliname "id" stulpeli ir eilutes su trukstamomis reiksmes
    data.drop("id", axis=1, inplace=True)
    data = data[data["bare_nuclei"] != "?"]
    # Konvertuojame visus stulpelius i skaiciu
    for col in data.columns:
        data[col] = pd.to_numeric(data[col])
    # Sumaisome duomenis
    data = data.sample(frac=1).reset_index(drop=True)
    return data


# 1.2 Padalinti duomenis i mokymo (80%), validavimo (10%) ir testavimo (10%) aibes.
def split_data(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    total = len(data)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    train = data.iloc[:train_end].reset_index(drop=True)
    val = data.iloc[train_end:val_end].reset_index(drop=True)
    test = data.iloc[val_end:].reset_index(drop=True)
    return train, val, test


# 2. MODELIO TRENIRAVIMAS, VALIDAVIMAS IR TESTAVIMAS
# 2.1 Sigmoidine aktyvacijos funkcija
def sigmoid_function(a):
    return 1 / (1 + np.exp(-a))


# 2.2 Ivertinam modeli
def evaluate_model(data, w):
    features = data.columns.drop("class")
    predictions = []
    true_labels = []
    error_sum = 0
    for _, row in data.iterrows():
        # Pridedame poslinkio reiksme (1) i požymiu vektoriu
        x = np.array([1] + [row[f] for f in features])
        a = np.dot(w, x)
        y = sigmoid_function(a)
        prediction = round(y)
        predictions.append(prediction)
        # Konvertuojame originalia klase: 2 -> 0 ir 4 -> 1
        true_label = row["class"] / 2 - 1
        true_labels.append(true_label)
        error_sum += (true_label - y) ** 2
    accuracy = np.mean(np.array(predictions) == np.array(true_labels))
    return error_sum, accuracy, predictions, true_labels


# 2.3 Treniruojam neurona fiksuojant mokymo ir validavimo klaidas bei tiksluma kiekvienoje epochoje.
def trainBatchWithMetrics(
    train_data, val_data, errorsMin, learningRate=0.05, epochs=2000
):
    features = train_data.columns.drop("class")
    w = np.zeros(
        len(features) + 1
    )  # Inicijuojame svorius ir poslinki (w[0] yra poslinkis)
    train_errors = []
    train_accuracies = []
    val_errors = []
    val_accuracies = []

    for epoch in range(epochs):
        data = train_data.sample(frac=1)
        totalError = 0
        gradientSum = np.zeros(len(features) + 1)
        # Sukaupti gradientai per visus mokymo pavyzdzius
        for _, row in data.iterrows():
            x = np.array([1] + [row[f] for f in features])
            a = np.dot(w, x)
            y = sigmoid_function(a)
            t = (
                row["class"] / 2 - 1
            )  # Tikslinė reiksme: 0 (nepiktybinis) arba 1 (piktybinis)
            gradientSum += (y - t) * y * (1 - y) * x
            totalError += (t - y) ** 2
        # Atnaujiname svorius naudodami vidutini gradienta
        w = w - learningRate * (gradientSum / len(data))
        train_errors.append(totalError)
        _, train_acc, _, _ = evaluate_model(train_data, w)
        train_accuracies.append(train_acc)
        val_error, val_acc, _, _ = evaluate_model(val_data, w)
        val_errors.append(val_error)
        val_accuracies.append(val_acc)
        # print(
        #     f"Paketinio GD epocha {epoch}: mokymo klaida = {totalError:.4f}, tikslumas = {train_acc:.4f}, validavimo klaida = {val_error:.4f}, tikslumas = {val_acc:.4f}"
        # )
        # Jei mokymo klaida maziau uz nustatyta riba, stabdome mokyma
        if totalError < errorsMin:
            print(
                "Paketinio gradientinio nusileidimo: stabdomas mokymas epochoje", epoch
            )
            epoch += 1  # Skaiciuojame si epocha kaip baigta
            break
    return w, epoch, train_errors, train_accuracies, val_errors, val_accuracies


# 2.4 Funkcija "trainStochasticWithMetrics": treniruoja neurona naudojant stochastini gradientini nusileidima,
#     fiksuojant mokymo ir validavimo klaidas bei tiksluma kiekvienoje epochoje.
def trainStochasticWithMetrics(
    train_data, val_data, errorsMin, learningRate=0.05, epochs=10000
):
    features = train_data.columns.drop("class")
    w = np.zeros(len(features) + 1)
    train_errors = []
    train_accuracies = []
    val_errors = []
    val_accuracies = []

    for epoch in range(epochs):
        data = train_data.sample(frac=1)
        totalError = 0
        # Atnaujiname svorius kiekvienam pavyzdziui atskirai
        for _, row in data.iterrows():
            x = np.array([1] + [row[f] for f in features])
            a = np.dot(w, x)
            y = sigmoid_function(a)
            t = row["class"] / 2 - 1
            for j in range(len(w)):
                w[j] = w[j] - learningRate * (y - t) * y * (1 - y) * x[j]
            totalError += (t - y) ** 2
        train_errors.append(totalError)
        _, train_acc, _, _ = evaluate_model(train_data, w)
        train_accuracies.append(train_acc)
        val_error, val_acc, _, _ = evaluate_model(val_data, w)
        val_errors.append(val_error)
        val_accuracies.append(val_acc)
        # print(
        #     f"Stochastinio GD epocha {epoch}: mokymo klaida = {totalError:.4f}, tikslumas = {train_acc:.4f}, validavimo klaida = {val_error:.4f}, tikslumas = {val_acc:.4f}"
        # )
        if totalError < errorsMin:
            print(
                "Stochastinio gradientinio nusileidimo: stabdomas mokymas epochoje",
                epoch,
            )
            epoch += 1
            break
    return w, epoch, train_errors, train_accuracies, val_errors, val_accuracies


# 3. TYRIMU ATLIKIMAS IR REZULTATU VIZUALIZACIJA
if __name__ == "__main__":
    # 3.1 Ikeliame ir paruosime duomenis
    data = prepare_data("breast-cancer.data")
    print("Duomenu irasu skaicius po valymo:", len(data))

    # 3.2 Padaliname duomenis i mokymo (80%), validavimo (10%) ir testavimo (10%) aibes
    train_set, val_set, test_set = split_data(data, 0.8, 0.1, 0.1)
    print(
        "Mokymo aibe:",
        len(train_set),
        "Validavimo aibe:",
        len(val_set),
        "Testavimo aibe:",
        len(test_set),
    )

    # 3.3 Eksperimentas 1: Mokymas naudojant paketini gradientini nusileidima
    print("\nMokymas naudojant paketinio gradientinio nusileidimo metoda...")
    start_time = time.time()
    (
        w_batch,
        epochs_batch,
        train_err_batch,
        train_acc_batch,
        val_err_batch,
        val_acc_batch,
    ) = trainBatchWithMetrics(
        train_set, val_set, errorsMin=12, learningRate=0.05, epochs=500
    )
    batch_time = time.time() - start_time
    test_err_batch, test_acc_batch, test_pred_batch, test_true_batch = evaluate_model(
        test_set, w_batch
    )

    print("\nRezultatai naudojant paketini gradientini nusileidima:")
    print("Galutiniai svoriai ir poslinkis:", w_batch)
    print("Epochos skaicius:", epochs_batch)
    print(
        "Paskutines epochos mokymo klaida:",
        train_err_batch[-1],
        "Mokymo tikslumas:",
        train_acc_batch[-1],
    )
    print(
        "Paskutines epochos validavimo klaida:",
        val_err_batch[-1],
        "Validavimo tikslumas:",
        val_acc_batch[-1],
    )
    print("Testavimo klaida:", test_err_batch, "Testavimo tikslumas:", test_acc_batch)
    print("Mokymo laikas (s):", batch_time)

    # Isspausdiname prognozes kiekvienam testavimo pavyzdziui
    print("\nPaketinio GD: Testavimo pavyzdziu prognozes:")
    for i, (pred, true_val) in enumerate(zip(test_pred_batch, test_true_batch)):
        print(f"Pavyzdys {i+1}: Prognoze = {pred}, Tiesa = {int(true_val)}")

    # Pavaizduojame mokymo ir validavimo klaidu priklausomybe nuo epochu (paketinis metodas)
    plt.figure()
    plt.plot(range(len(train_err_batch)), train_err_batch, label="Mokymo klaida")
    plt.plot(range(len(val_err_batch)), val_err_batch, label="Validavimo klaida")
    plt.xlabel("Epochos")
    plt.ylabel("Klaida (kvadratiniu klaidu suma)")
    plt.title("Paketinis GD: Klaida nuo epochu")
    plt.legend()
    plt.show()

    # Pavaizduojame mokymo ir validavimo tikslumo priklausomybe nuo epochu (paketinis metodas)
    plt.figure()
    plt.plot(range(len(train_acc_batch)), train_acc_batch, label="Mokymo tikslumas")
    plt.plot(range(len(val_acc_batch)), val_acc_batch, label="Validavimo tikslumas")
    plt.xlabel("Epochos")
    plt.ylabel("Tikslumas")
    plt.title("Paketinis GD: Tikslumas nuo epochu")
    plt.legend()
    plt.show()

    # 3.4 Eksperimentas 2: Mokymas naudojant stochastini gradientini nusileidima
    print("\nMokymas naudojant stochastini gradientini nusileidima...")
    start_time = time.time()
    (
        w_stoch,
        epochs_stoch,
        train_err_stoch,
        train_acc_stoch,
        val_err_stoch,
        val_acc_stoch,
    ) = trainStochasticWithMetrics(
        train_set,
        val_set,
        errorsMin=12,
        learningRate=0.05,
        epochs=10000,
    )
    stoch_time = time.time() - start_time
    test_err_stoch, test_acc_stoch, test_pred_stoch, test_true_stoch = evaluate_model(
        test_set, w_stoch
    )

    print("\nRezultatai naudojant stochastini gradientini nusileidima:")
    print("Galutiniai svoriai ir poslinkis:", w_stoch)
    print("Epochos skaicius:", epochs_stoch)
    print(
        "Paskutines epochos mokymo klaida:",
        train_err_stoch[-1],
        "Mokymo tikslumas:",
        train_acc_stoch[-1],
    )
    print(
        "Paskutines epochos validavimo klaida:",
        val_err_stoch[-1],
        "Validavimo tikslumas:",
        val_acc_stoch[-1],
    )
    print("Testavimo klaida:", test_err_stoch, "Testavimo tikslumas:", test_acc_stoch)
    print("Mokymo laikas (s):", stoch_time)

    # Isspausdiname prognozes kiekvienam testavimo pavyzdziui (prognozuota vs. tikra klase)
    print("\nStochastinio GD: Testavimo pavyzdziu prognozes:")
    for i, (pred, true_val) in enumerate(zip(test_pred_stoch, test_true_stoch)):
        print(f"Pavyzdys {i+1}: Prognoze = {pred}, Tiesa = {int(true_val)}")

    # Pavaizduojame mokymo ir validavimo klaidu priklausomybe nuo epochu (stochastinis metodas)
    plt.figure()
    plt.plot(range(epochs_stoch), train_err_stoch, label="Mokymo klaida")
    plt.plot(range(epochs_stoch), val_err_stoch, label="Validavimo klaida")
    plt.xlabel("Epochos")
    plt.ylabel("Klaida (kvadratiniu klaidu suma)")
    plt.title("Stochastinis GD: Klaida nuo epochu")
    plt.legend()
    plt.show()

    # Pavaizduojame mokymo ir validavimo tikslumo priklausomybe nuo epochu (stochastinis metodas)
    plt.figure()
    plt.plot(range(epochs_stoch), train_acc_stoch, label="Mokymo tikslumas")
    plt.plot(range(epochs_stoch), val_acc_stoch, label="Validavimo tikslumas")
    plt.xlabel("Epochos")
    plt.ylabel("Tikslumas")
    plt.title("Stochastinis GD: Tikslumas nuo epochu")
    plt.legend()
    plt.show()

    # 3.5 Eksperimentas 3: Itaka mokymosi greiciui (paketinis metodas)
    learning_rates = [0.01, 0.05, 0.1]
    batch_results = {}

    print("\nIvertiname skirtingus mokymosi greicius (paketinis metodas)...")
    # Paleidziame eksperimenta su skirtingomis mokymosi greicio reiksmes
    for lr in learning_rates:
        start = time.time()
        w_tmp, epochs_tmp, train_err_tmp, train_acc_tmp, val_err_tmp, val_acc_tmp = (
            trainBatchWithMetrics(
                train_set,
                val_set,
                errorsMin=12,
                learningRate=lr,
                epochs=500,
            )
        )
        elapsed = time.time() - start
        test_err_tmp, test_acc_tmp, _, _ = evaluate_model(test_set, w_tmp)
        batch_results[lr] = {
            "epochs": epochs_tmp,
            "train_error": train_err_tmp[-1],
            "train_accuracy": train_acc_tmp[-1],
            "val_error": val_err_tmp[-1],
            "val_accuracy": val_acc_tmp[-1],
            "test_error": test_err_tmp,
            "test_accuracy": test_acc_tmp,
            "time": elapsed,
        }
        print(
            f"Mokymosi greitis {lr}: Testavimo tikslumas = {test_acc_tmp:.4f}, Mokymo laikas = {elapsed:.4f} s"
        )

    # Stulpeline diagrama, palyginanti testavimo tiksluma skirtingiems mokymosi greiciams (paketinis metodas)
    plt.figure()
    plt.bar(
        [str(lr) for lr in learning_rates],
        [batch_results[lr]["test_accuracy"] for lr in learning_rates],
    )
    plt.xlabel("Mokymosi greitis")
    plt.ylabel("Testavimo tikslumas")
    plt.title("Paketinis GD: Testavimo tikslumas nuo mokymosi greicio")
    plt.show()
