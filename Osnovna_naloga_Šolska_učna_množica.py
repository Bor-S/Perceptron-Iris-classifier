# Osnovna
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

# Leave-One-Out validacija
for i in range(len(X)):
    # Pripravimo podatke
    X_train = np.delete(X, i, axis=0)
    Y_train = np.delete(Y, i, axis=0)
    X_test = X[i]
    Y_test = Y[i]

    # Negiramo U2 samo pri treniranju
    X_mod_train = prepare_data(X_train, Y_train, negate_U2=True)
    X_mod_test = prepare_data(np.array([X_test]), np.array([Y_test]), negate_U2=False)
    
    weights = perceptron_train(X_mod_train)
    Y_pred = perceptron_test(weights, X_mod_test[0])

    print(f"Test {i + 1}: Pravi razred: {Y_test}, Napovedani razred: {Y_pred}, Uteži: {weights}")
