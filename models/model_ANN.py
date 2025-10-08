import scipy.special
import numpy as np
class NeuralNetwork:
    def __init__(self,input_nodes,hidden_nodes,output_nodes,learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.weight_input_hidden = np.random.normal(0.0,pow(self))