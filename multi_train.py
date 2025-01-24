import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os 

def unit_nn(input_shape, output_shape, units, layers_per_segment):
    input_layer = tf.keras.layers.Input(shape=input_shape)
    x = input_layer
    for _ in range(layers_per_segment):
        x = tf.keras.layers.Dense(units, activation='tanh')(x)
    output_layer = tf.keras.layers.Dense(output_shape, activation= 'tanh')(x)
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    return model

def multi_shot(x, y, layers_per_segment, units, epochs, learning_rate):
    # Randomly initialize trainable middle states
    middle_state0 = tf.Variable(tf.random.normal([x.shape[0], 1], dtype=tf.float32), trainable=True)
    middle_state1 = tf.Variable(tf.random.normal([x.shape[0], 1], dtype=tf.float32), trainable=True)

    # Instantiate each segment
    segment1 = unit_nn(x.shape[1:], 16, units=units, layers_per_segment=layers_per_segment) # shape x = (1, ), y = 16
    segment2 = unit_nn(16, 16, units=units, layers_per_segment=layers_per_segment) # shape x = (16, ), y = 16
    segment3 = unit_nn(16, y.shape[1], units=units, layers_per_segment=layers_per_segment) # shape x = (16, ), y = 1

    # Combine segments into a single unified model
    input_layer = tf.keras.layers.Input(shape=x.shape[1:])
    intermediate1 = segment1(input_layer)
    intermediate2 = segment2(intermediate1)
    output_layer = segment3(intermediate2)
    full_model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    optimizer = tf.keras.optimizers.Adam(learning_rate)
    mse_loss = tf.keras.losses.MeanSquaredError()


    loss_record = []

    for epoch in range(epochs):
        # Train segment 1
        with tf.GradientTape() as tape1:
            intermediate1_output = segment1(x, training=True)
            loss1 = mse_loss(tf.reduce_sum(intermediate1_output, axis= -1), middle_state0)
        gradients1 = tape1.gradient(loss1, segment1.trainable_variables + [middle_state0])
        optimizer.apply_gradients(zip(gradients1, segment1.trainable_variables + [middle_state0]))

        # Train segment 2
        with tf.GradientTape() as tape2:
            intermediate2_output = segment2(intermediate1_output, training=True)
            loss2 = mse_loss(tf.reduce_sum(intermediate2_output, axis= -1), middle_state1)
        gradients2 = tape2.gradient(loss2, segment2.trainable_variables + [middle_state1])
        optimizer.apply_gradients(zip(gradients2, segment2.trainable_variables + [middle_state1]))

        # Train segment 3
        with tf.GradientTape() as tape3:
            predictions = segment3(intermediate2_output, training=True)
            loss3 = mse_loss(predictions, y)
        gradients3 = tape3.gradient(loss3, segment3.trainable_variables)
        optimizer.apply_gradients(zip(gradients3, segment3.trainable_variables))

        segements_loss = loss1 + loss2 + loss3
        full_loss = mse_loss(full_model(x), y)
        loss_record.append([segements_loss, full_loss])
        print(f"Epoch {epoch + 1}/{epochs}, Loss of segements: {segements_loss.numpy()}, Full model loss: {full_loss.numpy()}")

    return full_model, loss_record

def plot(x, y, predictions, name, loss: True):
    if loss:
        plt.plot(x, np.log10(y), label=name[0])
        plt.plot(x, np.log10(predictions), label=name[1])
    else:
        plt.plot(x, y, label=name[0])
        plt.plot(x, predictions, label=name[1])
    plt.legend()
    plt.pause(1)
    if not os.path.exists("images"):
        os.makedirs("images")
    plt.savefig(f"images/{name[2]}.png")
    plt.cla()

def main():
    x = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1).astype(np.float32) 
    y = np.sin(x).reshape(-1, 1).astype(np.float32)  

    model, loss_record = multi_shot(
        x, y,
        layers_per_segment=2,
        units=16,
        epochs=2000,
        learning_rate=0.001
    )

    predictions = model.predict(x)
    plot(x, y, predictions,
         ["True sin(x)", "Predicted sin(x)", "results"], loss=False)
    plot(range(len(loss_record)), 
         [record[0] for record in loss_record], 
         [record[1] for record in loss_record], 
         ["segements loss", "full model loss", "losses"], loss=True)

if __name__ == "__main__":
    main()