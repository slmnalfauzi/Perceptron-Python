import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_derivative(y_true, y_pred):
    return -(y_true - y_pred)

X = np.array([[0, 0], 
              [0, 1], 
              [1, 0], 
              [1, 1]])  # 4 sampel, masing-masing 2 input
y = np.array([[0], [1], [1], [0]])  # Target output (XOR problem)

# Parameter Jaringan Saraf
input_layer_neurons = 2    # Jumlah neuron input (x1, x2)
hidden_layer_neurons = 4   # Jumlah neuron hidden layer
output_layer_neurons = 1   # Jumlah neuron output

np.random.seed(42)
weights_input_hidden = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))
bias_hidden = np.random.uniform(size=(1, hidden_layer_neurons))
weights_hidden_output = np.random.uniform(size=(hidden_layer_neurons, output_layer_neurons))
bias_output = np.random.uniform(size=(1, output_layer_neurons))


learning_rate = 0.1
epochs = 10000

for epoch in range(epochs):
    # FORWARD PROPAGATION
    hidden_layer_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)
    
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    output_layer_output = sigmoid(output_layer_input)
    
    # LOSS (Error)
    loss = mse(y, output_layer_output)
    
    # BACKWARD PROPAGATION
    # Hitung gradien untuk output layer
    d_output = mse_derivative(y, output_layer_output) * sigmoid_derivative(output_layer_output)
    error_hidden_layer = d_output.dot(weights_hidden_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)
    
    # Update bobot dan bias
    weights_hidden_output -= hidden_layer_output.T.dot(d_output) * learning_rate
    bias_output -= np.sum(d_output, axis=0, keepdims=True) * learning_rate
    weights_input_hidden -= X.T.dot(d_hidden_layer) * learning_rate
    bias_hidden -= np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate
    
    # Print loss setiap 1000 epoch
    if epoch % 1000 == 0:
        print(f"Epoch {epoch} Loss: {loss}")

# Prediksi hasil akhir
print("\nPrediksi setelah training:")
hidden_layer_input = np.dot(X, weights_input_hidden) + bias_hidden
hidden_layer_output = sigmoid(hidden_layer_input)
output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
output_layer_output = sigmoid(output_layer_input)
print(output_layer_output)
