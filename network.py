import tensorflow as tf
import numpy as np
from tqdm import tqdm
tf.set_random_seed(1997)

class FeedforwardNet:
    def __init__(self, data, flags):
        self.network = None
        self.FLAGS = flags
        self.train_data = tf.data.Dataset.from_tensor_slices(
            (tf.cast(data[0][0]/255, tf.float32), tf.cast(data[0][1], tf.int64))
        ).shuffle(2000).batch(self.FLAGS.batch_size)
        self.test_data = tf.data.Dataset.from_tensor_slices(
            (tf.cast(data[1][0] / 255, tf.float32), tf.cast(data[1][1], tf.int64))
        ).batch(self.FLAGS.batch_size)
        self.input_shape = data[0][0][0].shape
        self.samples = data[0][0].shape[0]
		
    def build_graph(self):
        """
        Builds the 5 layer network.
        """
        self.network = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=self.input_shape),
            tf.keras.layers.Dense(1000, activation=tf.nn.relu, use_bias=False),
            tf.keras.layers.Dense(1000, activation=tf.nn.relu, use_bias=False),
            tf.keras.layers.Dense(500, activation=tf.nn.relu, use_bias=False),
            tf.keras.layers.Dense(200, activation=tf.nn.relu, use_bias=False),
            tf.keras.layers.Dense(10, use_bias=False)
        ])

    def train(self):
        optimizer = tf.train.AdamOptimizer()
        global_step = tf.train.get_or_create_global_step()
        epoch_acc = list()
        for epoch in range(self.FLAGS.epochs):
            accuracies = list()
            for x, y in tqdm(self.train_data, total=round(self.samples/self.FLAGS.batch_size)):
                with tf.GradientTape() as tape:
                    output = self.network(x)
                    loss = tf.losses.softmax_cross_entropy(tf.one_hot(y, 10), output)
                gradient = tape.gradient(loss, self.network.trainable_weights)
                optimizer.apply_gradients(zip(gradient, self.network.trainable_weights), global_step)
                correct = tf.equal(tf.argmax(output, axis=1, output_type=tf.int64), y)
                accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
                accuracies.append(accuracy)
            print("Epoch: ", epoch, "Accuracy: ", np.mean(accuracies))
            epoch_acc.append(np.mean(accuracies))
        return epoch_acc