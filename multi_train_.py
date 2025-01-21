import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# Define the Multi-Shooting Model
class MultiShootModel(tf.keras.Model):
    def __init__(self, num_segments, layers_per_segment, units):
        super(MultiShootModel, self).__init__()
        self.num_segments = num_segments
        self.segments = []
        for _ in range(num_segments): # 5 units each layer
            segment = tf.keras.Sequential([
                tf.keras.layers.Dense(units, activation='relu') for _ in range(layers_per_segment)
            ])
            self.segments.append(segment)
        self.initial_layer = tf.keras.layers.Dense(units, activation='relu')
        self.final_layer = tf.keras.layers.Dense(1)

    def call(self, x, middle_states):
        assert len(middle_states) == num_segments, f"the number of middle states {len(middle_states)}, \n,
                                                    don't match with the number of segment {num_segments}"
        middle_train = []
        x_segment = self.initial_layer(x)
        for i, middle_state in enumerate(middle_states):
            middle_state = self.initial_layer(middle_state)
            middle_train.append(self.segments[i](middle_state))

        for segment in self.segments:
            x_segment = segment(x_segment)
        output = self.final_layer(x_segment)
        return output, middle_train

# Continuity loss
def middle_state_loss(segment_outputs, middle_state, middle_state_shape):
    loss = 0
    for i in range(len(segment_outputs) - 1):
        if middle_state_shape[-1] == 2:
            loss += tf.reduce_mean(tf.square(segment_outputs[i][:, -1] - segment_outputs[i + 1][:, 0])) # The last unit of the first segment's second layer and the first element of the next segment's first layer are set to be equal.
        ## use a simple network to init all middle layers, then use those middle states as deeper network's middle states
        elif middle_state_shape[-1] == 5:
            loss += tf.reduce_mean(tf.square(segment_outputs[i] - middle_state[i])) 
        elif middle_state_shape[-1] == 1:
            loss += tf.reduce_mean(tf.square(tf.reduce_sum(segment_outputs[i], axis= -1) - middle_state[i]))
        else:
            raise ValueError('middle_state_shape[-1] should be 1 or 2 or 5')
    return loss

def training(x, 
             y, 
             num_segments, 
             layers_per_segment, 
             units, 
             epochs, 
             learning_rate,
             middle_state_shape,
             name):
    # Instantiate Model
    model = MultiShootModel(num_segments, layers_per_segment, units)
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    if middle_state_shape[-1]:
        middle_state = np.random.rand(middle_state_shape[0], middle_state_shape[1]).astype(np.float32) # 1 or number of units
        middle_state = [tf.Variable(middle_state)] * num_segments

    # Training Loop
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            # Forward pass
            y_pred, segment_outputs = model(x)

            # Compute fit loss
            fit_loss = tf.reduce_mean(tf.square(y - y_pred))
            if middle_state_shape[-1]:
                # Middle state loss
                mid_loss = middle_state_loss(segment_outputs, middle_state, middle_state_shape)

                # Total loss
                total_loss = fit_loss + mid_loss
            else:
                total_loss = fit_loss

        # Backpropagation
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Print progress
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Fit Loss: {fit_loss.numpy():.4f}, Total Loss: {total_loss.numpy():.4f}")
    return model, total_loss.numpy()

def save_img(x, y, y_pred, name, loss, total_time):
    plt.figure()   
    plt.plot(x, y, label='True')
    plt.plot(x, y_pred, label='Predicted')
    plt.text(0, 0, f'Loss: {loss:.4f} Total_Time:{total_time:.4f}', fontsize=12)
    plt.legend()
    if not os.path.exists('./images'):
        os.makedirs('./images')
    plt.savefig(f'./images/{name}.png')
    plt.pause(1)

def main(args):
    x = args['x']
    y = args['y']
    start = time.perf_counter()
    model, loss = training(**args)
    total_time = time.perf_counter() - start
    y_pred, _ = model(x)
    name = args['name']
    save_img(x, y, y_pred, name, loss, total_time)


if __name__ == '__main__':
    x = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1).astype(np.float32)
    y = np.sin(x).astype(np.float32)

    # Hyperparameters
    num_segments = 3
    layers_per_segment = 2
    units = 5
    epochs = 1000
    learning_rate = 1e-3
    times = 3
    middle_state_shape = (len(x), 2) # 0: without middle state, 
                                     # 1: with middle state shape n*1, 
                                     # 2: special case, connecting the last unit and first unit of next segement, 
                                     # 5: with middle state shape n*5
    for j in range(times):  
        args = {'x': x,
                'y': y,
                'num_segments': num_segments, 
                'layers_per_segment': layers_per_segment, 
                'units': units, 
                'epochs': epochs, 
                'learning_rate': learning_rate,
                'middle_state_shape': middle_state_shape,
                'name': f'middle_state_shape_{middle_state_shape}_epochs_{epochs}_learning_rate_{learning_rate}_times_{j}'}
        main(args)

