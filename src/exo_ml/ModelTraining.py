from tensorflow.keras.regularizers import l1, l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
import numpy as np
from create_model import create_model
from functools import partial
from ExtractDataForTraining import compute_data_for_training
from KeyPointsExtractions import KeyPointsExtractions
from CurvesReconstruction import CurveReconstruction
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

TF_ENABLE_ONEDNN_OPTS = 0

dataset_file = "processed_data.h5"

key_points_extraction = KeyPointsExtractions()
curve_reconstruction = CurveReconstruction()

age_max = 25
gender = 1
joint = "knee"
direction = "x"
curve_repetition = 50

conditions = {
    "Age": None,
    "Gender": None
}
attributes = ["Age", "Gender", "Height", "Weight"]
X_inputs, y_data = compute_data_for_training(dataset_file=dataset_file, joint=joint, direction=direction, curve_repetition=curve_repetition, attributes_to_include=attributes, conditions_on_attributes=conditions)

X_inputs_array = np.array(X_inputs)
y_data_array = np.array(y_data)

assert len(X_inputs) == len(y_data), "Mismatch between X_inputs and y_data lengths"

# Load your dataset and preprocess it
# Example:
# Step 1: Split data into training and the remainder (validation + test)
X_temp, X_test, y_temp, y_test = train_test_split(X_inputs_array, y_data_array, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_temp = scaler.fit_transform(X_temp)
X_test = scaler.transform(X_test)

# Step 2: Split the remainder into validation and test sets
# X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
# Note: 0.25 * 0.8 = 0.2, so validation and test sets are both 20% of the total data

input_shape = X_temp.shape[1]

create_model_with_input_shape = partial(create_model, input_shape=input_shape, number_outputs=24)

# Wrap the model using the function you created
model_ = KerasRegressor(model=create_model_with_input_shape, verbose=0)

# Define the grid search parameters
param_grid = {
    'batch_size': [30, 50],
    'epochs': [250, 300, 350],
    'model__learning_rate': [0.001, 0.008],
    'model__neurons': [64, 128, 256],
    'model__dropout_rate': [0.0, 0.2],
    'model__activation': ['relu', 'elu', "prelu", "leakyrelu"],
    'model__regularizer': [None, l1(0.01), l2(0.01)],  # Add L1 and L2 regularizers
    'model__initializer': ['he_normal', 'glorot_uniform'],
    'model__number_of_hidden_layers': [3, 4, 5, 6],
}

# Create GridSearchCV
grid = GridSearchCV(estimator=model_, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3, verbose=2)

# Fit the grid search
grid_result = grid.fit(X_temp, y_temp)

# Summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, std, param in zip(means, stds, params):
    relative_std_dev = std / abs(mean)
    print("MSE: %f (Std: %f, Relative Std: %f) with: %r" % (mean, std, relative_std_dev, param))
    if relative_std_dev < 0.2:
        print("The standard deviation is within the acceptable range.")
    else:
        print("The standard deviation indicates high variability.")

# Extract the best model hyperparameters
best_params = grid_result.best_params_
# best_params = {
#     'batch_size': 30,
#     'epochs': 150,
#     'model__learning_rate': 0.002,
#     'model__neurons': 128,
#     'model__dropout_rate': 0.0,
#     'model__activation': 'relu',
#     'model__regularizer': None
# }
print("Best parameters found: ", best_params)

# Create the best model
best_model = create_model_with_input_shape(learning_rate=best_params['model__learning_rate'],
                                           activation=best_params['model__activation'],
                                           neurons=best_params['model__neurons'],
                                           dropout_rate=best_params['model__dropout_rate'],
                                           regularizer=best_params['model__regularizer'])

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

# Train the model with the best parameters and record the history
history = best_model.fit(X_train, y_train,
                         epochs=best_params['epochs'],
                         batch_size=best_params['batch_size'],
                         validation_data=(X_val, y_val),
                         verbose=2)

# Plot the training and validation loss
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Training and Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid(True)
# plt.savefig("1st_model_results.png", dpi=300)
# plt.show()

test_loss, test_mae = best_model.evaluate(X_test, y_test)
print(f'Test Mean Absolute Error: {test_mae}')
print(f'Test Loss: {test_loss}')

