# Adaptive Neural Network Controller
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers.schedules import ExponentialDecay

# ... (data loading, preprocessing, plant, and controller code as before) ...
# Generate random inputs within a specific range
# You can manually enter the input here
def get_next_input():
    input_data = np.array([0, 0, 70, 10, -50])  # Modify this line to your desired initial conditions
    return input_data.reshape(1, -1)

# You can manually enter the desired coordinates here
def get_desired_coordinates():
    desired_coordinates = np.array([15, -10 , 5])  # Modify this line to your desired coordinates
    return desired_coordinates.reshape(1, -1)



# Controller
controller = Sequential([
    Dense(256, activation='relu', input_shape=(3,)),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(5)
])

#optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
#controller.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
# ExponentialDecay learning rate schedule
initial_learning_rate = 1e-2
decay_steps = 50
decay_rate = 0.95
min_learning_rate = 1e-6

lr_schedule = ExponentialDecay(
    initial_learning_rate,
    decay_steps=decay_steps,
    decay_rate=decay_rate,
    staircase=True,
    name='ExponentialDecayLearningRate'
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
#optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule)

controller.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
# Replace the optimizer in your controller with the new learning rate schedule
#optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Online training loop
num_iterations = 500
online_batch_size = 1
errors = []
window_size = 10
tolerance = 1e-3


errors_x = []
errors_y = []
errors_z = []
import matplotlib.pyplot as plt

# ... (all the setup code as before)

plant_outputs = []

for i in range(num_iterations):
    input_sample = get_next_input()
    input_sample = np.array(input_sample).reshape(1, -1)
    input_sample = input_scaler.transform(input_sample)

    current_coordinates = plant.predict(input_sample)
    desired_coordinates = get_desired_coordinates()

    with tf.GradientTape() as tape:
        tape.watch(controller.trainable_variables)
        control_signal = controller(desired_coordinates)
        control_signal = tf.reshape(control_signal, (1, 5))

        plant_output = plant(control_signal)

        # Record the output of the plant for this iteration
        plant_outputs.append(plant_output.numpy()[0])

        error = tf.reduce_mean(tf.square(desired_coordinates - plant_output))
        if error < 0.01:
            print(f"Stopping training at iteration {i} with error {error.numpy()}")
            break
    gradients = tape.gradient(error, controller.trainable_variables)
    optimizer.apply_gradients(zip(gradients, controller.trainable_variables))

    error_x = (desired_coordinates[0][0] - plant_output[0][0])
    error_y = (desired_coordinates[0][1] - plant_output[0][1])
    error_z = (desired_coordinates[0][2] - plant_output[0][2])

    errors_x.append(error_x.numpy())
    errors_y.append(error_y.numpy())
    errors_z.append(error_z.numpy())
    errors.append(error.numpy())

# ... (all the error plot code as before)

# Plot of the plant output over time
plant_outputs = np.array(plant_outputs)
plt.figure(figsize=(10, 5))
plt.plot(plant_outputs[:, 0], label='Output X')
plt.plot(plant_outputs[:, 1], label='Output Y')
plt.plot(plant_outputs[:, 2], label='Output Z')
plt.xlabel('Iteration')
plt.ylabel('Plant Output')
plt.legend()
plt.show()



plt.figure(figsize=(10, 5))
plt.plot(errors_x, label='Error X')
plt.plot(errors_y, label='Error Y')
plt.plot(errors_z, label='Error Z')
plt.plot(errors, label='Error ')
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.legend()
plt.show()

# Rest of the code...

# Separate plots for each error
plt.figure(figsize=(10, 5))
plt.plot(errors_x, label='Error X',color='red')
plt.xlabel('Iteration')
plt.ylabel('Error X')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(errors_y, label='Error Y',color='blue')
plt.xlabel('Iteration')
plt.ylabel('Error Y')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(errors_z, label='Error Z', color='green')
plt.xlabel('Iteration')
plt.ylabel('Error Z')
plt.legend()
plt.show()

# A plot for the total error
plt.figure(figsize=(10, 5))
plt.plot(errors, label='Total Error',color='purple')
plt.xlabel('Iteration')
plt.ylabel('Total Error')
plt.legend()
plt.show()




    

   
