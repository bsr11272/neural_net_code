import streamlit as st
import numpy as np
import tensorflow as tf
from models.forward_propagation import create_model, perform_forward_propagation, visualize_network, display_layer_calculations
from models.backward_propagation import calculate_partial_derivatives, generate_latex_formula
import streamlit.components.v1 as components

# Declare the custom React component
interactive_nn_component = components.declare_component("interactive_nn_component", url="http://localhost:3001")

st.set_page_config(layout="wide")

def main():
    st.title("Interactive Neural Network Explorer with TensorFlow")

    # Sidebar for network configuration
    st.sidebar.header("Network Configuration")
    num_layers = st.sidebar.slider("Number of Layers", 2, 10, 3)

    layers = []
    activations = []
    for i in range(num_layers):
        layers.append(st.sidebar.number_input(f"Nodes in Layer {i+1}", 1, 100, 5))
        if i < num_layers - 1:
            activations.append(st.sidebar.selectbox(f"Activation for Layer {i+1}", ["relu", "sigmoid", "tanh"], key=f"act_{i}"))
        else:
            activations.append("linear")  # Output layer typically has a linear activation

    model = create_model(layers, activations)
    st.session_state['model'] = model

    # Visualize the network using the matplotlib/networkx method
    if 'on_back_prop_page' not in st.session_state:
        plt_buf = visualize_network(layers, plt_show=False)
        st.image(plt_buf)

    # Handle input and propagation
    handle_input_and_propagation(layers, num_layers)

    # Debug: Print session state
    st.write("Debug: Final session state keys:", list(st.session_state.keys()))
    st.write("Debug: on_back_prop_page:", st.session_state.get('on_back_prop_page', False))

    # Check if we should display the backward propagation page and React component
    if st.session_state.get('on_back_prop_page', False):
        st.write("Debug: Attempting to display backward propagation page")
        display_backward_propagation_page()
    else:
        st.write("Debug: Not on backward propagation page")

# def handle_input_and_propagation(layers, num_layers):
#     st.header("Input")
#     input_choice = st.radio("Input Type", ["User Defined", "Random"])
#     input_values = None

#     if input_choice == "User Defined":
#         user_input = st.text_input("Enter input values (comma-separated)")
#         if user_input:
#             try:
#                 input_values = np.array([float(x.strip()) for x in user_input.split(",")]).reshape(1, -1)
#             except ValueError:
#                 st.error("Invalid input. Please enter comma-separated numbers.")
#                 return
#     else:
#         input_values = np.random.rand(1, layers[0])
#         st.write("Random Input:", input_values.flatten())

#     col1, col2 = st.columns(2)
#     with col1:
#         if st.button("Perform Forward Propagation"):
#             st.write("Debug: Forward propagation button clicked")
#             output, error = perform_forward_propagation(st.session_state['model'], input_values)
#             if error:
#                 st.error(error)
#             else:
#                 st.write("Network Output:", output)
#                 st.session_state['input_values'] = input_values
#                 st.session_state['forward_prop_done'] = True
#                 st.write("Debug: Forward propagation completed")

#     with col2:
#         if st.button("Prepare for Backward Propagation"):
#             st.write("Debug: Backward propagation preparation clicked")
#             if 'input_values' in st.session_state:
#                 st.session_state['prepare_backprop'] = True
#                 st.session_state['on_back_prop_page'] = True  # Set flag for backward propagation page
#                 st.experimental_rerun()
#             else:
#                 st.error("Please perform forward propagation first.")
#                 st.write("Debug: Input values not found in session state")

#     if st.session_state.get('prepare_backprop', False):
#         st.write("Debug: Preparing for backward propagation")
#         with st.form("backward_prop_form"):
#             true_values = st.text_input("Enter true output values (comma-separated):")
#             submit_button = st.form_submit_button("Perform Backward Propagation")
            
#             if submit_button:
#                 st.write("Debug: Backward propagation form submitted")
#                 if true_values:
#                     try:
#                         true_values = np.array([float(x.strip()) for x in true_values.split(",")]).reshape(1, -1)
#                         st.session_state['true_values'] = true_values
#                         st.experimental_rerun()
#                     except ValueError:
#                         st.error("Invalid input. Please enter comma-separated numbers.")
#                 else:
#                     st.error("Please enter true output values.")

#     st.write("Debug: Current session state keys:", list(st.session_state.keys()))
#     st.write("Debug: on_back_prop_page:", st.session_state.get('on_back_prop_page', False))

#     if 'forward_prop_done' in st.session_state and 'on_back_prop_page' not in st.session_state:
#         display_detailed_calculations(num_layers, layers)

def handle_input_and_propagation(layers, num_layers):
    st.header("Input")
    input_choice = st.radio("Input Type", ["User Defined", "Random"])
    input_values = None

    if input_choice == "User Defined":
        user_input = st.text_input("Enter input values (comma-separated)")
        if user_input:
            try:
                input_values = np.array([float(x.strip()) for x in user_input.split(",")]).reshape(1, -1)
            except ValueError:
                st.error("Invalid input. Please enter comma-separated numbers.")
                return
    else:
        input_values = np.random.rand(1, layers[0])
        st.write("Random Input:", input_values.flatten())

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Perform Forward Propagation"):
            st.write("Debug: Forward propagation button clicked")
            output, error = perform_forward_propagation(st.session_state['model'], input_values)
            if error:
                st.error(error)
            else:
                st.write("Network Output:", output)
                st.session_state['input_values'] = input_values
                st.session_state['forward_prop_done'] = True
                st.write("Debug: Forward propagation completed")

    with col2:
        if st.button("Prepare for Backward Propagation"):
            st.write("Debug: Backward propagation preparation clicked")
            if 'input_values' in st.session_state:
                st.session_state['prepare_backprop'] = True
                st.experimental_rerun()
            else:
                st.error("Please perform forward propagation first.")
                st.write("Debug: Input values not found in session state")

    if st.session_state.get('prepare_backprop', False):
        st.write("Debug: Preparing for backward propagation")
        with st.form("backward_prop_form"):
            true_values = st.text_input("Enter true output values (comma-separated):")
            submit_button = st.form_submit_button("Perform Backward Propagation")
            
            if submit_button:
                st.write("Debug: Backward propagation form submitted")
                if true_values:
                    try:
                        true_values = np.array([float(x.strip()) for x in true_values.split(",")]).reshape(1, -1)
                        st.session_state['true_values'] = true_values
                        st.session_state['on_back_prop_page'] = True
                        st.session_state['backprop_started'] = True  # New flag to start backward propagation visualization
                        st.write("Debug: Session state updated for backward propagation")
                        st.experimental_rerun()
                    except ValueError:
                        st.error("Invalid input. Please enter comma-separated numbers.")
                else:
                    st.error("Please enter true output values.")

    st.write("Debug: Current session state keys:", list(st.session_state.keys()))
    st.write("Debug: on_back_prop_page:", st.session_state.get('on_back_prop_page', False))

    if 'forward_prop_done' in st.session_state and 'on_back_prop_page' not in st.session_state:
        display_detailed_calculations(num_layers, layers)



def display_detailed_calculations(num_layers, layers):
    st.header("Detailed Layer Calculations")
    selected_layer = st.selectbox("Select Layer", range(1, num_layers))
    layer_idx = selected_layer - 1
    if layer_idx < len(layers) - 1:
        weights = st.session_state['model'].layers[layer_idx].get_weights()[0].T
        biases = st.session_state['model'].layers[layer_idx].get_weights()[1].reshape(-1, 1)
        activation_func = st.session_state['model'].layers[layer_idx].activation
        inputs = st.session_state['input_values'] if layer_idx == 0 else st.session_state['model'].layers[layer_idx - 1](st.session_state['input_values']).numpy()
        display_layer_calculations(layer_idx, weights, biases, inputs, activation_func, st)

# def display_backward_propagation_page():
#     st.header("Backward Propagation Visualization")
    
#     # Show the custom React component only when backward propagation starts
#     clicked_element = interactive_nn_component(layers={"layers": st.session_state['model'].layers})

#     if clicked_element:
#         st.write(f"Clicked Element: {clicked_element}")
#         # You can perform additional calculations or visualizations for backpropagation here.

#     fig = visualize_network_with_clicks(
#         st.session_state['model'],
#         st.session_state['input_values'],
#         st.session_state['true_values']
#     )
#     st.plotly_chart(fig)

#     st.subheader("Backpropagation Formula")
#     layer = st.selectbox("Select Layer", range(1, len(st.session_state['model'].layers) + 1))
#     param_type = st.radio("Select Parameter Type", ["Weight", "Bias"])
    
#     if param_type == "Weight":
#         max_value = st.session_state['model'].layers[layer-1].units - 1 if layer > 1 else st.session_state['model'].input_shape[1] - 1
#         i = st.number_input("Select input neuron index", min_value=0, max_value=max_value, value=0)
#         j = st.number_input("Select output neuron index", min_value=0, max_value=st.session_state['model'].layers[layer-1].units-1, value=0)
#         formula = display_backprop_formula(st.session_state['model'], layer-1, 'kernel', i, j)
#     else:
#         max_value = st.session_state['model'].layers[layer-1].units - 1 if layer > 1 else st.session_state['model'].input_shape[1] - 1
#         i = st.number_input("Select neuron index", min_value=0, max_value=max_value, value=0)
#         formula = display_backprop_formula(st.session_state['model'], layer-1, 'bias', i)
    
#     st.latex(formula)

def display_backward_propagation_page():
    # Check if backpropagation has started before displaying the visualization
    if st.session_state.get('backprop_started', False):
        st.header("Backward Propagation Visualization")

        # Debugging to ensure that the logic is being reached
        st.write("Debug: Backpropagation visualization started.")

        # Extract the number of neurons from each layer and pass that to the React component
        layer_sizes = [layer.units for layer in st.session_state['model'].layers]
        
        # Show the custom React component for neural network interaction
        clicked_element = interactive_nn_component(layers={"layers": layer_sizes})

        # Debugging to verify if the component is working
        st.write("Debug: React component loaded with layer sizes:", layer_sizes)

        if clicked_element:
            st.write(f"Clicked Element: {clicked_element}")
            
            # Extract information from the clicked element (e.g., which weight or bias was clicked)
            element_type, layer_idx, neuron_info = parse_clicked_element(clicked_element)
            
            if element_type in ['Weight', 'Bias']:
                i, j = neuron_info if element_type == 'Weight' else (neuron_info, None)
                
                # Generate the backpropagation formula using the `generate_latex_formula` function
                formula = generate_latex_formula(st.session_state['model'], layer_idx, 'w' if element_type == 'Weight' else 'b', i, j)
                st.latex(formula)

                # Calculate and display partial derivatives (gradients) using `calculate_partial_derivatives`
                partials = calculate_partial_derivatives(
                    st.session_state['model'],
                    st.session_state['input_values'],
                    st.session_state['true_values']
                )
                gradient_value = partials.get(clicked_element, "Not found")
                st.write(f"The gradient value for {clicked_element} is: {gradient_value}")
        else:
            st.write("Debug: No element clicked yet.")


# Helper function to parse the clicked element (React component)
def parse_clicked_element(element_name):
    """
    This helper function parses the clicked element name to extract relevant information.
    Example of expected format: 'Weight_Layer3_Neuron23' or 'Bias_Layer2_Neuron1'
    """
    parts = element_name.split('_')
    element_type = parts[0]  # 'Weight' or 'Bias'
    layer_idx = int(parts[1][5:]) - 1  # Extract layer index from 'LayerX'
    if element_type == 'Weight':
        i, j = int(parts[2][6:]), int(parts[2][7:])
        return element_type, layer_idx, (i, j)
    else:
        i = int(parts[2][6:])
        return element_type, layer_idx, i


if __name__ == "__main__":
    main()
