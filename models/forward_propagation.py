import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import networkx as nx
from typing import List
from io import BytesIO
import random

def create_model(layers: List[int], activations: List[str]) -> tf.keras.Sequential:
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(layers[0],)))
    
    for i in range(1, len(layers)):
        model.add(tf.keras.layers.Dense(layers[i], activation=activations[i-1].lower()))
    
    return model

def perform_forward_propagation(model, input_values):
    if input_values.shape[1] != model.input_shape[1]:
        return None, f"Input size ({input_values.shape[1]}) does not match the first layer size ({model.input_shape[1]})"
    model_output = model(input_values)
    return model_output.numpy().flatten(), None

def visualize_network(layers: List[int], plt_show=True):
    G = nx.DiGraph()
    pos = {}
    node_colors = []
    node_sizes = []

    for i, layer_size in enumerate(layers):
        layer_name = "I" if i == 0 else "H" if i < len(layers) - 1 else "O"
        for j in range(layer_size):
            node_id = f"{layer_name}{i+1}_{j+1}"
            G.add_node(node_id)
            pos[node_id] = (i, -j)
            node_colors.append("lightblue" if layer_name == "I" else "lightgreen" if layer_name == "O" else "lightgray")
            node_sizes.append(1000)

    for i in range(len(layers) - 1):
        for j in range(layers[i]):
            for k in range(layers[i+1]):
                G.add_edge(f"{'I' if i == 0 else 'H'}{i+1}_{j+1}", 
                           f"{'O' if i == len(layers) - 2 else 'H'}{i+2}_{k+1}")

    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=node_sizes, 
            font_size=8, font_weight='bold', arrows=True)
    plt.title("Neural Network Architecture")
    if plt_show:
        plt.show()
    else:
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        return buf

def display_layer_calculations(layer_idx: int, weights, biases, inputs, activation_func, st=None):
    if st:
        st.write(f"### Layer {layer_idx + 1} Calculations")
    else:
        print(f"### Layer {layer_idx + 1} Calculations")

    inputs = inputs.T  # Ensure correct shape for dot product

    for node_idx in range(weights.shape[0]):
        if st:
            st.write(f"#### Node {node_idx + 1}")
        else:
            print(f"#### Node {node_idx + 1}")

        z = np.dot(weights[node_idx], inputs) + biases[node_idx]
        a = activation_func(z)

        if st:
            st.latex(r"z = \sum_{i} w_i x_i + b")
            st.latex(f"z = {np.round(z, 4)}")
            st.latex(f"a = f(z) = {np.round(a, 4)}")
        else:
            print(f"z = {np.round(z, 4)}")
            print(f"a = {np.round(a, 4)}")
    
        generate_node_image_with_networkx(layer_idx, node_idx, weights, biases, inputs.T, z, a, st)


def generate_node_image_with_networkx(layer_idx: int, node_idx: int, weights, biases, inputs, z, a, st):
    G = nx.DiGraph()
    pos = {}
    node_colors = []
    edge_colors = []

    random_bias = random.uniform(-1, 1)  # Random bias between -1 and 1
    biases[node_idx] = random_bias  # Set random bias for the current node

    num_inputs = len(inputs.flatten())
    vertical_spacing = 2 / max(num_inputs, 1)  # Normalizing vertical space

    for i, x in enumerate(inputs.flatten()):
        input_node_label = f"x_{i+1}\n ({x:.3f})"
        G.add_node(input_node_label)
        pos[input_node_label] = (-1, -i * vertical_spacing)
        node_colors.append('#85C1E9')  # Light blue

    z_label = f"Z={z[0]:.3f}"
    G.add_node(z_label)
    pos[z_label] = (0, -vertical_spacing * (num_inputs - 1) / 2)
    node_colors.append('#76D7C4')  # Light mint green

    for i, w in enumerate(weights[node_idx]):
        input_node_label = f"x_{i+1}\n ({inputs.flatten()[i]:.3f})"
        G.add_edge(input_node_label, z_label, weight=f"w={w:.3f}")
        edge_colors.append('#34495E')  # Dark blue-grey

    bias_node_label = f"b={random_bias:.3f}"
    G.add_node(bias_node_label)
    pos[bias_node_label] = (-1, -vertical_spacing * (num_inputs))
    node_colors.append('#F7DC6F')  # Yellow for bias
    G.add_edge(bias_node_label, z_label, weight="bias")

    output_node_label = f"a_{node_idx+1}\n ({a[0]:.3f})"
    G.add_node(output_node_label)
    pos[output_node_label] = (1, -vertical_spacing * (num_inputs - 1) / 2)
    G.add_edge(z_label, output_node_label, weight="σ(Z)")
    node_colors.append('#82E0AA')  # Light green
    edge_colors.append('#34495E')  # Dark blue-grey

    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=5000, font_size=16, font_weight='bold', edge_color=edge_colors, arrows=True, arrowstyle='-|>', arrowsize=20)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

    plt.axis('off')
    plt.title(f"Layer {layer_idx + 1} - Node {node_idx + 1} Computations", size=20, color='#196F3D')

    if st:
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        st.image(buf)
    else:
        plt.show()