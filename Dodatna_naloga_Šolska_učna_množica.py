# Dodatna
import numpy as np

# Priprava podatkov
def prepare_data(X, Y, negate_U2=True):
    X_mod = []
    for i in range(len(X)):
        x_transformed = np.append(X[i], 1)  
        if Y[i] == 1 and negate_U2:  # Razred U2 pomnožimo z -1
            x_transformed = -x_transformed
        X_mod.append(x_transformed)
    X_mod = np.array(X_mod)
    return X_mod

# Treniranje perceptrona
def perceptron_train(X, epoch=4):
    w = np.array([-1, 0, 0])  # Začetni vektor koeficientov
    for _ in range(epoch):
        for x in X:
            if np.dot(w, x) <= 0:
                w = w + x  # Posodobi uteži
    return w

# Testiranje perceptrona
def perceptron_test(w, x):
    if np.dot(w, x) > 0:
        return 0  # U1
    else:
        return 1  # U2

# Učna množica
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([0, 0, 1, 1])  # U1: 0, U2: 1

# Usposabljanje perceptronov za pare razredov
def train_perceptrons(X, Y):
    num_classes = len(np.unique(Y))
    perceptrons = []

    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            X_train_pair = X[(Y == i) | (Y == j)]
            Y_train_pair = Y[(Y == i) | (Y == j)]

            X_mod_train_pair = prepare_data(X_train_pair, Y_train_pair, negate_U2=True)
            weights = perceptron_train(X_mod_train_pair)
            perceptrons.append((i, j, weights))

    return perceptrons

# Testiranje ločilne meje za pare razredov
def test_perceptrons(perceptrons, X_test):
    class_votes = [0, 0]
    
    for perceptron in perceptrons:
        i, j, weights = perceptron
        X_mod_test = prepare_data(np.array([X_test]), np.array([i]), negate_U2=False)
        result = perceptron_test(weights, X_mod_test[0])
        
        if result == 0:
            class_votes[i] += 1
        else:
            class_votes[j] += 1

    predicted_class = np.argmax(class_votes)
    return predicted_class

# Usposabljanje perceptronov za pare razredov
perceptrons = train_perceptrons(X, Y)

# Testiranje z vzorci
for i in range(len(X)):
    X_test = X[i]
    Y_test = Y[i]

    predicted_class = test_perceptrons(perceptrons, X_test)

    # Izpis rezultatov
    print(f"Test {i + 1}: Pravi razred: {Y_test}, Napovedani razred: {predicted_class}, Uteži: {weights}")