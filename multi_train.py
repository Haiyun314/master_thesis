import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os 
import time

def unit_nn(input_shape, output_shape, units, layers_per_segment):
    input_layer = tf.keras.layers.Input(shape=input_shape)
    x = input_layer
    for _ in range(layers_per_segment):
        x = tf.keras.layers.Dense(units, activation='tanh')(x)
    output_layer = tf.keras.layers.Dense(output_shape, activation= 'tanh')(x)
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    return model

def multi_shot(x, y, layers_per_segment, number_of_segments, units, epochs, learning_rate, train_full_model):
    # Randomly initialize trainable middle states
    middle_state = [tf.Variable(tf.random.normal([x.shape[0], units], dtype=tf.float32), trainable=True) for _ in range(number_of_segments)]
    all_states = [x] + middle_state + [y]
    segments = []
    # Instantiate each segment
    for i in range(0, len(all_states) - 1):
        segment = unit_nn(all_states[i].shape[1:], all_states[i + 1].shape[1], units=units, layers_per_segment=layers_per_segment)
        segments.append(segment)

    # Combine segments into a single unified model
    input_layer = tf.keras.layers.Input(shape=x.shape[1:])
    middle_segment = input_layer
    for element in segments:
        middle_segment = element(middle_segment)
    output_layer = middle_segment
    full_model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    optimizer = tf.keras.optimizers.Adam(learning_rate)
    mse_loss = tf.keras.losses.MeanSquaredError()

    loss_record = []

    for epoch in range(epochs):
        if train_full_model:
            start_timer = time.perf_counter()
            with tf.GradientTape() as tape:
                full_loss = mse_loss(full_model(x), y)
            gradients = tape.gradient(full_loss, full_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, full_model.trainable_variables))
            end_timer = time.perf_counter()
            del tape
            time_consume = end_timer - start_timer
            loss_record.append([time_consume, full_loss])
            print(f'full_loss={full_loss}, time comsume = {time_consume}, epoch = {epoch} \n')
        else:
            start_timer = time.perf_counter()
            segments_loss = 0
            for i in range(0, len(segments)):
                with tf.GradientTape() as tape:
                    intermediate_output = segments[i](all_states[i], training=True)
                    loss = mse_loss(intermediate_output, all_states[i + 1])

                # Note: the place of middle_state in the trainable variables list is important
                if i == len(segments) - 1:
                    gradients = tape.gradient(loss, segments[i].trainable_variables)
                    optimizer.apply_gradients(zip(gradients, segments[i].trainable_variables))
                else:
                    gradients = tape.gradient(loss, segments[i].trainable_variables + [middle_state[i]])
                    optimizer.apply_gradients(zip(gradients, segments[i].trainable_variables + [middle_state[i]]))
                segments_loss += loss
                del tape
            end_timer = time.perf_counter()
            time_consume = end_timer - start_timer
            full_loss = mse_loss(full_model(x), y)
            loss_record.append([segments_loss, full_loss])
            print(f"Epoch {epoch + 1}/{epochs}, Loss of segments: {segments_loss.numpy()}, Full model loss: {full_loss.numpy()}, time_consume : {time_consume} \n")

    return full_model, loss_record

def plot(x, y, predictions, name, loss: bool = True):
    if loss: 
        plt.plot(x, np.log(y), label=name[0])
        plt.plot(x, np.log(predictions), label=name[1])
    else:
        plt.plot(x, y, label=name[0])
        plt.plot(x, predictions, label=name[1])
    plt.legend()
    plt.pause(1)
    if not os.path.exists("images"):
        os.makedirs("images")
    plt.savefig(f"images/{name[2]}.png")
    plt.cla()

def main(args):
    model, loss_record = multi_shot(
        **args
    )
    predictions = model.predict(args['x'])
    plot(args['x'], args['y'], predictions,
         ["True sin(x)", "Predicted sin(x)", "train_full_results"], loss=False)
    plot(range(len(loss_record)), 
         [record[0] for record in loss_record], 
         [record[1] for record in loss_record], 
         ["segements loss", "full model loss", "train_full_losses"], loss=True)

if __name__ == "__main__":
    x = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1).astype(np.float32) 
    y = np.sin(x).reshape(-1, 1).astype(np.float32)  
    args = {
        'x' : x, 
        'y' : y,
        'layers_per_segment':2,
        'number_of_segments':2,
        'units':16,
        'epochs':5000,
        'learning_rate':0.001,
        'train_full_model': False
    }
    main(args)