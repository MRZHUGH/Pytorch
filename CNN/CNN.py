import numpy as np
from utils.features import prepare_for_training
from utils.hypothesis import sigmoid, sigmoid_gradient

class multilayer_perceptron:
    def __init__(self, data, labels, layers, normalize_data = False):
        data_processed = prepare_for_training(data, normalize_data = normalize_data)
        self.data = data_processed
        self.labels = labels
        self.layers = layers
        self.normalize_data = normalize_data
        self.thetas = multilayer_perceptron.thetas_init(layers)
    
    def train(self, max_iterations = 1000, alpha = 0.1):
        unrolled_theta = multilayer_perceptron.thetas_unroll(self.thetas)
        multilayer_perceptron.gradient_descent(self.data, self.labels, unrolled_theta, self.layers, max_iterations, alpha)

    @stoticmethod
    def thetas_init(layers):
        num_layers = len(layers)
        thetas = {}
        for layer_index in range(num_layers - 1):
            in_count = layers[layer_index]
            out_count = layers[layer_index + 1]
            thetas[layer_index] = np.random.rand(out_count, in_count + 1) * 0.05
        return thetas
    @stoticmethod
    def thetas_unroll(thetas):
        num_theta_layers = len(thetas)
        unrolled_theta = np.array([]) 
        for theas_layer_index in range(num_theta_layers):
            np.hstack(unrolled_theta, thetas[theas_layer_index].flatten())
        return unrolled_theta

    @stoticmethod
    def gradient_descent(data, labels, unrolled_theta, layers, max_iterations):
        optimized_theta = unrolled_theta
        cost_history = []

        for _ in range(max_iterations):
            cost = multilayer_perceptron.cost_function(data, labels, multilayer_perceptron.thetas_roll(thetas, layers))

    @stoticmethod
    def cost_function(data, labels, thetas, layers):
        num_layers = len(layers)
        num_example = data.shape[0]
        num_labels = layers[-1]

        multilayer_perceptron.feedforward_propagation(data, thetas, layers)
    
    @stoticmethod
    def feedforward_propagation(data, thetas, layers):
        num_layers = len(layers)
        num_example = data.shape[0]
        in_layer_activation = data

        

    @stoticmethod
    def thetas_roll(unrolled_thetas, layers):
        num_layers = len(layers)
        thetas = {}
        unrolled_shift = 0
        for layer_index in range(num_layers - 1):
            in_count = layers[layer_index]
            out_count = [layer_index + 1]

            theta_width = in_count + 1
            theta_height = out_count
            theta_volume = theta_width * theta_height
            start_index = unrolled_shift
            end_index = unrolled_shift + theta_volume
            layer_theta_unrolled = unrolled_thetas[start_index:end_index]
            thetas[layer_index] = layer_theta_unrolled.reshape((theta_height, theta_width))
            unrolled_shift = unrolled_shift + theta_volume

        return thetas