import tensorflow as tf
import numpy as np

# Function to compute gradients using TensorFlow
def compute_gradients(model, input_values, true_values):
    with tf.GradientTape() as tape:
        predictions = model(input_values, training=True)
        loss = tf.keras.losses.MeanSquaredError()(true_values, predictions)

    # Get gradients with respect to trainable variables
    gradients = tape.gradient(loss, model.trainable_variables)
    
    return gradients, loss

# Function to calculate partial derivatives (gradients) for each weight and bias
def calculate_partial_derivatives(model, input_values, true_values):
    gradients, _ = compute_gradients(model, input_values, true_values)
    
    partials = {}
    for layer_idx, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.layers.Dense):
            weights, biases = gradients[2*layer_idx], gradients[2*layer_idx + 1]
            for i in range(weights.shape[0]):
                for j in range(weights.shape[1]):
                    partials[f'w_{layer_idx+1}_{i+1}{j+1}'] = weights[i, j].numpy()
            for i in range(biases.shape[0]):
                partials[f'b_{layer_idx+1}_{i+1}'] = biases[i].numpy()
    
    return partials

# Function to generate the backpropagation path
def get_backprop_path(model, layer_idx, param_type, i, j=None):
    path = []
    current_layer = len(model.layers) - 1
    
    while current_layer >= layer_idx:
        if current_layer == layer_idx:
            if param_type == 'w':
                path.append(f'∂J/∂w^({current_layer+1})_{i+1}{j+1}')
            else:
                path.append(f'∂J/∂b^({current_layer+1})_{i+1}')
        else:
            path.append(f'∂a^({current_layer+1})/∂z^({current_layer+1})')
            path.append(f'∂z^({current_layer+1})/∂a^({current_layer})')
        current_layer -= 1
    
    return ' * '.join(reversed(path))

# Function to generate LaTeX backpropagation formula
def generate_latex_formula(model, layer_idx, param_type, i, j=None):
    formula = r"\frac{\partial J}{\partial "
    if param_type == 'w':
        formula += f"w^{{({layer_idx+1})}}_{{{i+1}{j+1}}}"
    else:
        formula += f"b^{{({layer_idx+1})}}_{{{i+1}}}"
    
    formula += "} = "
    
    path = get_backprop_path(model, layer_idx, param_type, i, j)
    terms = path.split(' * ')
    
    for idx, term in enumerate(terms):
        if idx > 0:
            formula += r" \cdot "
        formula += term
    
    return formula
