import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 1. Duomenų apdorojimas
def prepare_data(file_path):
    # Formatuojame failą į csv formatą
    prefix_line = "id,clump_thickness,uniform_cell_size,uniform_cell_shape,margin_adhesion,single_cell_size,bare_nuclei,bland_chromatin,normal_nucleoli,mitoses,class"
    with open(file_path, 'r+') as source:
        content = source.read()
        out = open("temp.data", "w")
        out.seek(0, 0)
        out.write(prefix_line.rstrip('\r\n') + '\n' + content)

    data = pd.read_csv("temp.data")
    os.remove("temp.data")
    # Pašaliname nereikalingus duomenis
    data.drop('id', axis=1, inplace=True)
    data.drop(data.index[(data['bare_nuclei'] == "?")], axis=0, inplace=True)
    # Permaisome duomenis
    data = data.sample(frac=1)
    print(data)


prepare_data("breast-cancer.data")

# 2. Dirbtinis neuronas
# Aktyvacijos funkcijos
def sigmoid_function(a):
    return 1 / (1 + np.exp(-a))


class Neuron:
    def __init__(self, w1, w2, b):
        self.w1 = w1
        self.w2 = w2
        self.bias = b

    def __predict_to_class(self, a):
        prediction = sigmoid_function(a)
        return round(prediction)

    # Gauti rezultatą rementis x1 ir x2 taškais, jų svoriais ir poslinkiu.
    # Pritaikoma paskirta aktyvacijos funkcija
    def predict(self, x1, x2):
        a = self.w1 * x1 + self.w2 * x2 + self.bias
        return self.__predict_to_class(a)


# 3. Rasti tinkamus svorių ir bias rinkinius (be mokymo; step funkcija)
def find_weight_sets(data, num_sets=3, max_iter=100000):
    found_sets = []
    iterations = 0

    # Ieškome svorių, kol nesurasime trijų rinkinių
    while len(found_sets) < num_sets and iterations < max_iter:
        w1 = np.random.uniform(low=-10, high=10)
        w2 = np.random.uniform(low=-10, high=10)
        b = np.random.uniform(low=-10, high=10)
        neuron = Neuron(w1, w2, b)

        # Patikriname, ar visi duomenų taškai teisingai klasifikuojami
        all_correct = True
        for _, row in data.iterrows():
            # Tikriname, ar su sugeneruotais svoriais tinkamai prognozuojami visi duomenys
            if neuron.predict(row['X'], row['Y']) != row['Class']:
                all_correct = False
                break

        # Jeigu visų taškų aktyvacijos funkcijos rezultatai atitinka jų klases,
        # pridedame rinkinį prie rezultatų
        if all_correct:
            found_sets.append((w1, w2, b))
        iterations += 1

    return found_sets


# 4. Rasti tinkamus svorių ir bias rinkinius (be mokymo; sigmoid funkcija)
weight_sets2 = find_weight_sets(data)
print("\nRasti tinkami svorių (w1, w2) ir bias (b) rinkiniai (sigmoidinė funkcija):")
for idx, (w1, w2, b) in enumerate(weight_sets2, start=1):
    print(f"Rinkinys {idx}: w1 = {w1:.4f}, w2 = {w2:.4f}, bias = {b:.4f}")
