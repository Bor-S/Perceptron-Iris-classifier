import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Nalaganje in priprava podatkov
def load_iris_data(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        data = [line.strip().split(',') for line in lines if line.strip()]
    X = np.array([list(map(float, row[:-1])) for row in data])
    Y = np.array([row[-1] for row in data])
    label_encoder = LabelEncoder()
    Y_encoded = label_encoder.fit_transform(Y)
    return X, Y_encoded

def prepare_data(X, Y, negate_U2=True):
    X_mod = []
    for i in range(len(X)):
        x_transformed = np.append(X[i], 1)  # Dodamo bias
        if Y[i] == 1 and negate_U2:  # Razred U2 pomnožimo z -1
            x_transformed = -x_transformed
        X_mod.append(x_transformed)
    return np.array(X_mod)

def perceptron_train_iris(X, epoch=10):
    w = np.array([-1, 0, 0, 0, 0])  # Začetni vektor koeficientov
    for _ in range(epoch):
        for x in X:
            if np.dot(w, x) <= 0:
                w = w + x
    return w

def perceptron_test(w, x):
    return 0 if np.dot(w, x) > 0 else 1

def standardize_data(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)

def evaluate_performance(Y_test, Y_pred):
    return np.sum(Y_test == Y_pred) / len(Y_test) * 100

# Treniranje perceptrona za posamezen razred
def train_perceptron_for_class(X, Y, class_index):
    Y_binary = np.where(Y == class_index, 1, 0)  # Treniranje za določen razred
    X_mod = prepare_data(X, Y_binary, negate_U2=False)
    weights = perceptron_train_iris(X_mod)
    return weights

# Treniranje perceptronov za vsak razred
def train_perceptrons_for_each_class(X, Y):
    num_classes = len(np.unique(Y))
    perceptrons = {}

    for i in range(num_classes):
        weights = train_perceptron_for_class(X, Y, i)
        perceptrons[i] = weights

    return perceptrons

# Razvrščanje z uporabo naučenih perceptronov
def classify_with_each_class_perceptrons(perceptrons, X):
    predictions = []
    for x in X:
        scores = [np.dot(w, np.append(x, 1)) for w in perceptrons.values()]
        predicted_class = np.argmax(scores)
        predictions.append(predicted_class)
    return predictions

# Nalaganje in priprava podatkov
X, Y = load_iris_data('iris.data')
X_standardized = standardize_data(X)

# Razdelitev podatkov
X_train, X_test, Y_train, Y_test = train_test_split(X_standardized, Y, test_size=1/3)

# Učenje perceptronov za vsak razred
perceptrons_each_class = train_perceptrons_for_each_class(X_train, Y_train)

# Klasifikacija in evalvacija
Y_pred = classify_with_each_class_perceptrons(perceptrons_each_class, X_test)
accuracy = evaluate_performance(Y_test, Y_pred)
print(f"Natančnost: {accuracy}%")
