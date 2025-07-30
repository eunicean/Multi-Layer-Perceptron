import numpy as np
import altair as alt 
import pandas as pd

def init_parameters(n_features, n_neurons, n_output): 
    """ TODO: mejorar explicación
    Parametros inciales de la red neuronal con valores aleatorios
        W1: Pesos de la capa de entrada 
        W2: Pesos de la capa a la de salida 
        b1: bias para la capa
        b2: bias para la capa de salida
    """

    np.random.seed(100)
    W1 = np.random.uniform(size = (n_features, n_neurons))
    b1 = np.random.uniform(size = (1, n_neurons))

    W2 = np.random.uniform(size = (n_neurons, n_output))
    b2 = np.random.uniform(size = (1, n_output))

    return {
        "W1" : W1 
        , "b1" : b1 
        , "W2" : W2
        , "b2" : b2 
    }

def linear_function(W, X, b): 
    """
    Calcula la transformación lineal XW + b
    """
    return (X @ W)+ b 

def sigmoid_func(Z): 
    """
    Función de activación sigmoide para una Zn
    """
    return 1 / (1 + np.exp(-Z))

def cost_function(A, y): 
    """
    Calcula la función de costo Binary Cross Entropy entre las predicciones A y los valores reales y
    """

    return -np.mean(y * np.log(A) + (1 - y) * np.log(1 - A))
    # https://www.geeksforgeeks.org/deep-learning/binary-cross-entropy-log-loss-for-binary-classification/


def predict(X, W1, W2, b1, b2): 
    """
    Predicción de la red neuronal
        1. Calcula la transformación lineal + activación para la capa oculta
        2. Calcula la transformación lineal + activación para la capa de salida
        3. Aplica umbral de 0.5 para obtener predicciones binarias

    """
    Z1 = linear_function(W1, X, b1)
    S1 = sigmoid_func(Z1) # a1
    Z2 = linear_function(W2, S1, b2)
    S2 = sigmoid_func(Z2) # a2
    return np.where(S2 >= 0.5, 1, 0)

def fit(X, y, n_features = 2, n_neurons = 3, n_output = 1, iterations = 10, eta = 0.001): 
    """
    Entrenamiento de la red neuronal 
       1. Calcula las activaciones
       2. Calcula el error
       3. Calcula gradientes y actualiza pesos
    """
    params = init_parameters(
        n_features= n_features
        , n_neurons= n_neurons
        , n_output= n_output
    )

    errors = [] 

    for _ in range(iterations): 
        """ Activaciones """
        Z1 = linear_function(params['W1'], X, params['b1'])
        S1 = sigmoid_func(Z1)
        Z2 = linear_function(params['W2'], S1, params['b2'])
        S2 = sigmoid_func(Z2)
        """ Cálculo de errores """
        error = cost_function(S2, y)
        errors.append(error)

        """ Claclulo de gradientes y nnuevos pesos """
        delta2 = (S2 - y) # derivada
        W2_gradients = S1.T @ delta2 
        params["W2"] = params["W2"] - W2_gradients * eta

        params["b2"] = params["b2"] - np.sum(delta2, axis = 0, keepdims= True) * eta 

        delta1 = (delta2 @ params["W2"].T) * S1 * (1 - S1)
        W1_gradients = X.T @ delta1 
        params["W1"] = params["W1"] - W1_gradients * eta 

        params["b1"] = params["b1"] - np.sum(delta1, axis = 0, keepdims= True) * eta 

    return errors, params 

y = np.array([[0, 1, 1, 0]]).T 
X = np.array([[0, 0, 1, 1]
              ,[0, 1, 0, 1]]).T 

errors, params = fit(X, y, iterations=5000, eta = 0.1)


y_pred = predict(X, params["W1"], params["W2"], params["b1"], params["b2"])
num_correct_predictions = (y_pred == y).sum()
accuracy = (num_correct_predictions / y.shape[0]) * 100
print('Multi-layer perceptron accuracy: %.2f%%' % accuracy)


alt.data_transformers.disable_max_rows()
df = pd.DataFrame({"errors":errors, "time-step": np.arange(0, len(errors))})
alt.Chart(df).mark_line().encode(x="time-step", y="errors").properties(title='Chart 2')
