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
                w = w + x # Posodobi uteži
    return w

def perceptron_test(w, x):
    return 0 if np.dot(w, x) > 0 else 1

def standardize_data(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)

def evaluate_performance(Y_test, Y_pred):
    return np.sum(Y_test == Y_pred) / len(Y_test) * 100

# Treniranje perceptrona za vsak par razredov
def train_perceptrons_for_pairs(X, Y):
    num_classes = len(np.unique(Y))
    perceptrons = {}

    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            X_pair = X[(Y == i) | (Y == j)]
            Y_pair = Y[(Y == i) | (Y == j)]
            Y_pair_binary = np.where(Y_pair == i, 0, 1)

            X_mod_pair = prepare_data(X_pair, Y_pair_binary)
            weights = perceptron_train_iris(X_mod_pair)
            perceptrons[(i, j)] = weights

    return perceptrons

# Razvrščanje z uporabo naučenih perceptronov
def classify_with_pairs(perceptrons, X, num_classes):
    predictions = []
    for x in X:
        votes = np.zeros(num_classes)
        for (i, j), w in perceptrons.items():
            result = perceptron_test(w, np.append(x, 1))
            if result == 0:  # U1
                votes[i] += 1
            else:  # U2
                votes[j] += 1
        predicted_class = np.argmax(votes)
        predictions.append(predicted_class)
    return predictions

# Nalaganje in priprava podatkov 
X, Y = load_iris_data('iris.data')
X_standardized = standardize_data(X)

# Razdelitev podatkov
X_train, X_test, Y_train, Y_test = train_test_split(X_standardized, Y, test_size=1/3)

# Učenje perceptronov za pare razredov
perceptrons_pairs = train_perceptrons_for_pairs(X_train, Y_train)

# Klasifikacija in evalvacija
Y_pred = classify_with_pairs(perceptrons_pairs, X_test, len(np.unique(Y)))
accuracy = evaluate_performance(Y_test, Y_pred)
print(f"Natančnost: {accuracy}%")
