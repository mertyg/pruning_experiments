import tensorflow as tf
import numpy as np
from tqdm import tqdm
tf.set_random_seed(1997)

class Experiment:
    def __init__(self, network: tf.keras.Sequential, data, prune_list):
        self.model = network
        self.data = data
        self.dense_weights = network.get_weights()
        self.prune_list = prune_list

    def tester(self, test_network=None):
        if test_network is None:
            test_network = self.model
        accuracy_list = list()
        for x, y in self.data:
            output = test_network(x)
            correct = tf.equal(tf.argmax(output, axis=1, output_type=tf.int64), y)
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
            accuracy_list.append(accuracy)
        return np.array(accuracy_list).mean()

    def restore(self):
        self.model.set_weights(self.dense_weights)

    def weight_pruning(self, k=25):
        weights = self.model.get_weights()
        pruned = list()
        for matrix in weights[:-1]:
            shape = matrix.shape
            flat = tf.reshape(matrix, (1, -1))
            threshold = np.percentile(tf.math.abs(flat), k)
            mask = tf.cast(tf.math.abs(flat) >= threshold, tf.float32)
            masked = tf.reshape(flat*mask, shape)
            pruned.append(masked)
        pruned.append(weights[-1])
        self.model.set_weights(pruned)
        accuracy = self.tester()
        self.restore()
        return accuracy

    def unit_pruning(self, k=25):
        weights = self.model.get_weights()
        pruned = list()
        for matrix in weights[:-1]:
            norms = tf.norm(matrix, ord=2, axis=0)
            threshold = np.percentile(norms, k)
            column_mask = norms >= threshold
            mask = tf.stack([column_mask for i in range(matrix.shape[0])])
            masked = tf.cast(mask, tf.float32)*matrix
            pruned.append(masked)
        pruned.append(weights[-1])
        self.model.set_weights(pruned)
        accuracy = self.tester()
        self.restore()
        return accuracy

    def prune_test(self, type="weight"):
        result = []
        for k in self.prune_list:
            if type == "weight":
                result.append(self.weight_pruning(k))
            else:
                result.append(self.unit_pruning(k))
        return result