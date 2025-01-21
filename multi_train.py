import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def unit_nn(input_shape, output_shape, units, layers_per_segment):
    input_layer = tf.keras.layers.Input(shape=input_shape)
    x = input_layer
    for _ in range(layers_per_segment):
        x = tf.keras.layers.Dense(units, activation='relu')(x)
    output_layer = tf.keras.layers.Dense(output_shape)(x)
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    return model

def multi_shot(x, y, layers_per_segment, units, epochs, learning_rate):
    # Randomly initialize trainable middle states
    middle_state0 = tf.Variable(tf.random.normal([x.shape[0], units], dtype=tf.float32), trainable=True)
    middle_state1 = tf.Variable(tf.random.normal([x.shape[0], units], dtype=tf.float32), trainable=True)

    # Instantiate each segment
    segment1 = unit_nn(x.shape[1:], middle_state0.shape[1], units=units, layers_per_segment=layers_per_segment)
    segment2 = unit_nn(middle_state0.shape[1:], middle_state1.shape[1], units=units, layers_per_segment=layers_per_segment)
    segment3 = unit_nn(middle_state1.shape[1:], y.shape[1], units=units, layers_per_segment=layers_per_segment)

    # Combine segments into a single unified model
    input_layer = tf.keras.layers.Input(shape=x.shape[1:])
    intermediate1 = segment1(input_layer)
    intermediate2 = segment2(intermediate1)
    output_layer = segment3(intermediate2)
    full_model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    optimizer = tf.keras.optimizers.Adam(learning_rate)
    mse_loss = tf.keras.losses.MeanSquaredError()

    for epoch in range(epochs):
        # Train segment 1
        with tf.GradientTape() as tape1:
            intermediate1_output = segment1(x, training=True)
            loss1 = mse_loss(intermediate1_output, middle_state0)
        gradients1 = tape1.gradient(loss1, segment1.trainable_variables + [middle_state0])
        optimizer.apply_gradients(zip(gradients1, segment1.trainable_variables + [middle_state0]))

        # Train segment 2
        with tf.GradientTape() as tape2:
            intermediate2_output = segment2(intermediate1_output, training=True)
            loss2 = mse_loss(intermediate2_output, middle_state1)
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
        print(f"Epoch {epoch + 1}/{epochs}, Loss of segements: {segements_loss.numpy()}, Full model loss: {full_loss.numpy()}")

    return full_model

def plot(x, y, predictions):
    plt.plot(x, y, label="True sin(x)")
    plt.plot(x, predictions, label="Predicted")
    plt.legend()
    plt.pause(1)

def main():
    x = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1).astype(np.float32) 
    y = np.sin(x).reshape(-1, 1).astype(np.float32)  

    model = multi_shot(
        x, y,
        layers_per_segment=2,
        units=16,
        epochs=1000,
        learning_rate=0.001
    )

    predictions = model.predict(x)
    plot(x, y, predictions)

if __name__ == "__main__":
    main()