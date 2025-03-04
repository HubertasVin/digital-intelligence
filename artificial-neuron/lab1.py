import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parametrai
center_0 = (2, 2)
center_1 = (-2, -2)
num_points = 10


# 1. Duomenų generavimas
def generate_data(center_0, center_1, num_points):
    np.random.seed(0)
    # Sugeneruojame duomenis abejoms klasėms aplinkui du centrus: (-2, -2) ir (2, 2)
    class_0_x = np.random.normal(loc=center_0[0], scale=0.5, size=num_points)
    class_0_y = np.random.normal(loc=center_0[1], scale=0.5, size=num_points)
    class_1_x = np.random.normal(loc=center_1[0], scale=0.5, size=num_points)
    class_1_y = np.random.normal(loc=center_1[1], scale=0.5, size=num_points)

    # Sujungiame sugeneruotus duomenis į patogesnį DataFrame tipą
    data = pd.DataFrame({
        'X': np.concatenate((class_0_x, class_1_x)),
        'Y': np.concatenate((class_0_y, class_1_y)),
        'Class': np.concatenate((np.zeros(num_points), np.ones(num_points)))
    })
    return data


data = generate_data(center_0, center_1, num_points)
print("Pradiniai duomenys su klasėmis:")
print(data)


# 2. Dirbtinis neuronas
# Aktyvacijos funkcijos
def step_function(a):
    return 1 if a >= 0 else 0


def sigmoid_function(a):
    return 1 / (1 + np.exp(-a))


class Perceptron:
    def __init__(self, w1, w2, b, activation_function):
        self.w1 = w1
        self.w2 = w2
        self.bias = b
        self.activation_function = activation_function

    def __predict_to_class(self, a):
        prediction = self.activation_function(a)
        # Jeigu rezultatas yra float tipo (panaudota sigmoid funkcija), apvaliname skaičių
        if (isinstance(prediction, float)):
            return round(prediction)
        else:
            return prediction

    # Gauti rezultatą rementis x1 ir x2 taškais, jų svoriais ir poslinkiu.
    # Pritaikoma paskirta aktyvacijos funkcija
    def predict(self, x1, x2):
        a = self.w1 * x1 + self.w2 * x2 + self.bias
        return self.__predict_to_class(a)


# 3. Rasti tinkamus svorių ir bias rinkinius (be mokymo; step funkcija)
def find_weight_sets(data, activation_function, num_sets=3, max_iter=100000, weight_range=(-10, 10)):
    found_sets = []
    iterations = 0

    # Ieškome svorių, kol nesurasime trijų rinkinių
    while len(found_sets) < num_sets and iterations < max_iter:
        w1 = np.random.uniform(*(-10, 10))
        w2 = np.random.uniform(*(-10, 10))
        b = np.random.uniform(*(-10, 10))
        neuron = Perceptron(w1, w2, b, activation_function)

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


weight_sets = find_weight_sets(data, step_function)
print("\nRasti tinkami svorių (w1, w2) ir bias (b) rinkiniai (step funkcija):")
for idx, (w1, w2, b) in enumerate(weight_sets):
    print(f"Rinkinys {idx + 1}: w1 = {w1:.4f}, w2 = {w2:.4f}, bias = {b:.4f}")


# 4. Rasti tinkamus svorių ir bias rinkinius (be mokymo; sigmoid funkcija)
weight_sets = find_weight_sets(data, sigmoid_function)
print("\nRasti tinkami svorių (w1, w2) ir bias (b) rinkiniai (sigmoidinė funkcija):")
for idx, (w1, w2, b) in enumerate(weight_sets, start=1):
    print(f"Rinkinys {idx}: w1 = {w1:.4f}, w2 = {w2:.4f}, bias = {b:.4f}")


# 5. Klases skiriančioji tiesė
def plot_data_and_boundaries(data, weight_sets):
    plt.figure(figsize=(8, 8))
    # Piešiame abiejų klasių taškus atskirai
    class_0 = data[data['Class'] == 0]
    class_1 = data[data['Class'] == 1]
    plt.scatter(class_0['X'], class_0['Y'], color='blue', label='Class 0')
    plt.scatter(class_1['X'], class_1['Y'], color='red', label='Class 1')

    colors = ['green', 'orange', 'purple']

    # Braižome kiekvieną sprendimo ribą
    for idx, (w1, w2, b) in enumerate(weight_sets):
        color = colors[idx % len(colors)]
        # Vaizduojame klases skiriančias tieses
        if abs(w2) < 1e-3:
            # Jeigu w2 yra arti 0, brėžiame vertikalią tiesę
            x_val = -b / w1
            plt.axvline(x=x_val, color=color,
                        label=f'Boundary {idx+1}: w1={w1:.2f}, w2={w2:.2f}, b={b:.2f}')
        else:
            # Apskaičiuojame tiesės galų kordinates taškuose -3 ir 3
            ys = [-(w1 * x + b) / w2 for x in [-3.0, 3.0]]
            plt.plot([-3, 3], ys, color=color,
                     label=f'Boundary {idx+1}: w1={w1:.2f}, w2={w2:.2f}, b={b:.2f}')

        # 6. Vektorių atvaizdavimas
        norm = np.sqrt(w1**2 + w2**2)
        if norm < 1e-6:
            continue
        # Atrandame tašką esantį ant tiesės
        x0 = -b * w1 / (norm**2)
        y0 = -b * w2 / (norm**2)

        # Normalizuojame svorių vektorių
        dx = (w1 / norm) * 2
        dy = (w2 / norm) * 2
        # Braižome rodyklę, kuri prasideda nuo taško ant sprendimo linijos
        plt.arrow(x0, y0, dx, dy, head_width=0.15, head_length=0.15, fc=color, ec=color)
        # Pažymime tašką, kuriame prasideda rodyklė
        plt.plot(x0, y0, 'o', color=color)

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Duomenys ir sprendimo ribos")
    plt.legend()
    # Nustatome ribas, kad grafiko X ir Y kraštinių santykis būtų 1:1.
    # Reikalinga, kad vektoriai matytūsi, jog jie yra statmeni jų tiesėms
    plt.gca().set_xlim([-4, 4])
    plt.gca().set_ylim([-4, 4])
    plt.show()


plot_data_and_boundaries(data, weight_sets)
