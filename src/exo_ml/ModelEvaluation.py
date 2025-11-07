import numpy as np
import tensorflow as tf
from create_model import create_model
from ExtractDataForTraining import compute_data_for_training, recompose_output, extract_data_to_train
from CurvesReconstruction import CurveReconstruction
from scipy.interpolate import make_interp_spline, splev
import matplotlib.pyplot as plt
import pandas as pd
import pickle

curve_reconstruction = CurveReconstruction()
plt.rcParams.update({'font.size': 16})
mean_relative_error = 0.0613


def compute_rmse(list1, list2):
    """
    Compute the Root Mean Square Error (RMSE) between two lists of the same length.

    Parameters:
    list1 (list or np.array): First list of numbers
    list2 (list or np.array): Second list of numbers

    Returns:
    float: The RMSE value
    """
    if len(list1) != len(list2):
        raise ValueError("Both lists must have the same length")

    list1 = np.array(list1)
    list2 = np.array(list2)

    mse = np.mean((list1 - list2) ** 2)
    rmse = np.sqrt(mse)
    return rmse


def are_lists_equal(list1, list2):
    if len(list1) != len(list2):
        return False
    for i in range(len(list1)):
        if list1[i] != list2[i]:
            return False
    return True


def is_ascending(lst):
    return lst == sorted(lst)


def list_rmse(x_input, y_real, y_predicted, attributes_included, attribute, parameter='relative_error'):
    length = len(x_input)
    error_on_gait_percentage = []
    error_on_joint_position = []
    corresponding_attributes_values = []
    for i in range(length):
        key_points_pos_real, key_points_values_real = recompose_output(y_real[i])
        key_points_pos_estimation, key_points_values_estimation = recompose_output(y_predicted[i])
        error_on_gait_percentage.append(compute_rmse(key_points_pos_real, key_points_pos_estimation))
        error = compute_rmse(key_points_values_real, key_points_values_estimation)
        # error = error * (180 / np.pi)
        if parameter == 'relative_error':
            amplitude = max(key_points_values_real) - min(key_points_values_real)
            error_on_joint_position.append(error / amplitude)
        else:
            error_on_joint_position.append(error)
        if attribute in attributes_included:
            index = attributes_included.index(attribute)
            corresponding_attributes_values.append(x_input[i][index])
        elif attribute == 'Speed':
            index = len(attributes_included)
            corresponding_attributes_values.append(x_input[i][index])
        else:
            print("No corresponding attributes")
            return 0

    return corresponding_attributes_values, error_on_joint_position, error_on_gait_percentage


def list_rmse_on_trajectories(inputs, y_traj, predicted_keypoints, real_keypoints, time_steps, attributes_included, attribute, parameter='relative_error'):
    error_on_predicted_traj = []
    corresponding_attributes_values = []
    number_of_false_cases = 0
    number_of_positive_cases = 0
    best_interpolation = 0
    worst_interpolation = 0
    typical_interpolation = 0
    best_error = 10
    best_keypoints = 0
    worst_error = 0
    worst_keypoints = 0
    typical_error = 0
    typical_keypoints = 0
    extrem_bad_case = 0
    for i in range(len(y_traj)):
        key_points_pos_estimation, key_points_values_estimation = recompose_output(predicted_keypoints[i])
        real_key_points_pos_estimation, real_key_points_values_estimation = recompose_output(real_keypoints[i])
        if is_ascending(key_points_pos_estimation):
            number_of_positive_cases += 1
            key_points_values_estimation = [x * (180 / np.pi) for x in key_points_values_estimation]
            real_key_points_values_estimation = [x * (180 / np.pi) for x in real_key_points_values_estimation]
            key_points_interpolation = curve_reconstruction.interpolation_sorted_values(sorted_pos=key_points_pos_estimation, sorted_values=key_points_values_estimation, interp_degree=3)
            time_norm = time_steps[i] / 100
            list_interp = key_points_interpolation(time_norm)
            root_mse = compute_rmse(y_traj[i], list_interp)
            if root_mse < best_error:
                best_error = root_mse
                best_interpolation = [time_norm, y_traj[i][:], list_interp]
                best_keypoints = [[real_key_points_pos_estimation, real_key_points_values_estimation], [key_points_pos_estimation, key_points_values_estimation]]
            if root_mse > worst_error:
                worst_error = root_mse
                worst_interpolation = [time_norm, y_traj[i][:], list_interp]
                worst_keypoints = [[real_key_points_pos_estimation, real_key_points_values_estimation],
                                  [key_points_pos_estimation, key_points_values_estimation]]
            if parameter == 'relative_error':
                amplitude = max(y_traj[i]) - min(y_traj[i])
                relative_root_mse = root_mse / amplitude
                error_on_predicted_traj.append(relative_root_mse)
                if relative_root_mse > 0.1:
                    extrem_bad_case += 1
                if mean_relative_error - 0.02 <= relative_root_mse <= mean_relative_error + 0.02:
                    typical_error = relative_root_mse
                    typical_interpolation = [time_norm, y_traj[i][:], list_interp]
                    typical_keypoints = [[real_key_points_pos_estimation, real_key_points_values_estimation],
                                       [key_points_pos_estimation, key_points_values_estimation]]
            else:
                error_on_predicted_traj.append(root_mse)
            if attribute in attributes_included:
                index = attributes_included.index(attribute)
                corresponding_attributes_values.append(inputs[i][index])
            elif attribute == 'Speed':
                index = len(attributes_included)
                corresponding_attributes_values.append(inputs[i][index])
            else:
                print("No corresponding attributes")
                return 0
        else:
            number_of_false_cases += 1
    if parameter == 'relative_error':
        amplitude_best = max(best_interpolation[1]) - min(best_interpolation[1])
        amplitude_worst = max(worst_interpolation[1]) - min(worst_interpolation[1])
        best_error = best_error / amplitude_best
        worst_error = worst_error / amplitude_worst

    return error_on_predicted_traj, corresponding_attributes_values, number_of_positive_cases, number_of_false_cases, best_error, best_interpolation, worst_error, worst_interpolation, extrem_bad_case, typical_error, typical_interpolation, best_keypoints, worst_keypoints, typical_keypoints


model_path = 'saved_model/my_regression_model_hip_abduction_all_population_new'
# Load the saved model
loaded_model = tf.keras.models.load_model(model_path)
print("Model loaded successfully.")

dataset_file = "processed_data.h5"
age_max = 25
gender = 1
joint = "hip"
direction = "y"
curve_repetition = 10

# conditions = {
#     "Age": [lambda x: x < age_max],
#     "Gender": [lambda x: x == gender]
# }
conditions_1 = {
    "Age": None,
    "Gender": None
}
attributes = ["Age", "Gender", "Height", "Weight"]
X_inputs, y_data, y_trajectories, timesteps = compute_data_for_training(dataset_file=dataset_file, joint=joint, direction=direction, curve_repetition=curve_repetition, attributes_to_include=attributes, conditions_on_attributes=conditions_1)

scaler_path = 'saved_scaler_hip_abduction_all_population_new.pkl'
# Load the scaler from the file
with open(scaler_path, 'rb') as f:
    loaded_scaler = pickle.load(f)
print("Scaler loaded successfully.")

# Transform the test data using the loaded scaler
X_inputs_scaled = loaded_scaler.transform(X_inputs)

# Predict using the loaded model
predictions = loaded_model.predict(X_inputs_scaled)


speed_values, errors_on_joint_position, errors_on_gait_percent = list_rmse(x_input=X_inputs, y_real=y_data, y_predicted=predictions, attributes_included=attributes, attribute="Speed", parameter='relative_error')
error_on_predicted_trajectories, spd_values, nbr_positive_cases, nbr_false_cases, min_error, best_interp, max_error, worst_interp, bad_cases, typical_err, typical_interp, bst_keypts, wrst_keypts, typical_keypts = list_rmse_on_trajectories(inputs=X_inputs, y_traj=y_trajectories, predicted_keypoints=predictions, real_keypoints=y_data,  time_steps=timesteps, attributes_included=attributes, attribute="Speed", parameter='relative_error')

# Convert lists to a pandas DataFrame
data_gait_percentage = pd.DataFrame({'speed': speed_values, 'gait_percentage_error': errors_on_gait_percent})
data_joint_position = pd.DataFrame({'speed': speed_values, 'joint_position_error': errors_on_joint_position})
data_error_on_predicted_trajectories = pd.DataFrame({'speed': spd_values, 'predicted_trajectories_error': error_on_predicted_trajectories})

# Group by abscissas and calculate mean and standard deviation
grouped_data_gait_percentage = data_gait_percentage.groupby('speed').agg(
    mean_ordinates=('gait_percentage_error', 'mean'),
    std_ordinates=('gait_percentage_error', 'std')
).reset_index()

grouped_data_joint_position = data_joint_position.groupby('speed').agg(
    mean_ordinates=('joint_position_error', 'mean'),
    std_ordinates=('joint_position_error', 'std')
).reset_index()

grouped_error_on_predicted_trajectories = data_error_on_predicted_trajectories.groupby('speed').agg(
    mean_ordinates=('predicted_trajectories_error', 'mean'),
    std_ordinates=('predicted_trajectories_error', 'std')
).reset_index()

print("Percentage of false cases:", (nbr_false_cases / (nbr_positive_cases + nbr_false_cases)) * 100)
print("Mean RMSE on predicted trajectories:", np.mean(error_on_predicted_trajectories))
print("Std RMSE on predicted trajectories:", np.std(error_on_predicted_trajectories))
print("Best error estimation:", min_error)
print("Worst error estimation:", max_error)
print("Typical error estimation:", typical_err)
print("Percentage of cases where the error on predicted traj > 10%:", bad_cases / nbr_positive_cases)
print(f'Mean RMSE on gait phase {np.mean(errors_on_gait_percent)}, STD {np.std(errors_on_gait_percent)}')
print(f'Mean relative RMSE on joint position {np.mean(errors_on_joint_position)}, STD {np.std(errors_on_joint_position)}')
print("Worst error on normalized gait phase:", max(errors_on_gait_percent))
print("Worst relative error on joint position:", max(errors_on_joint_position))

worst_real_interpolation = curve_reconstruction.interpolation_sorted_values(sorted_pos=wrst_keypts[0][0], sorted_values=wrst_keypts[0][1], interp_degree=3)
worst_real_interp = worst_real_interpolation(worst_interp[0])

# Plotting the mean and standard deviation
# plt.figure(1, figsize=(14, 6))
# plt.errorbar(grouped_data_gait_percentage['speed'], grouped_data_gait_percentage['mean_ordinates'],
#              yerr=grouped_data_gait_percentage['std_ordinates'], fmt='o', ecolor='r', capsize=5, label='Mean with Std Dev')
#
# plt.xlabel('Walking Speed (in m/s)')
# plt.ylabel('RMSE on normalized gait')
# plt.legend()
# plt.grid(True)
# plt.savefig("error_gait_percentage_hip_abduction_cp_new.png", dpi=600)
#
# plt.figure(2, figsize=(14, 6))
# plt.errorbar(grouped_data_joint_position['speed'], grouped_data_joint_position['mean_ordinates'],
#              yerr=grouped_data_joint_position['std_ordinates'], fmt='o', ecolor='r', capsize=5, label='Mean with Std Dev')
#
# plt.xlabel('Walking Speed (in m/s)')
# plt.ylabel('Relative RMSE on joint position')
# plt.legend()
# plt.grid(True)
# plt.savefig("relative_error_joint_position_hip_abduction_cp_new.png", dpi=600)
#

# plt.legend()# plt.figure(3, figsize=(14, 6))
# # plt.errorbar(grouped_error_on_predicted_trajectories['speed'], grouped_error_on_predicted_trajectories['mean_ordinates'],
# #              yerr=grouped_error_on_predicted_trajectories['std_ordinates'], fmt='o', ecolor='r', capsize=5, label='Mean with Std Dev')
# #
# # plt.xlabel('Walking Speed (in m/s)')
# # plt.ylabel('Relative RMSE on predicted trajectories')
# # # plt.ylim([-1, 3])
# # plt.legend()
# # plt.grid(True)
# # plt.savefig("relative_error_predicted_trajectories_hip_abduction_cp_new.png", dpi=600)
# #
plt.figure(4, figsize=(14, 6))
plt.scatter(bst_keypts[0][0], bst_keypts[0][1], label='Real Key points', color='blue')
plt.scatter(bst_keypts[1][0], bst_keypts[1][1], label='Predicted Key points', color='red')
plt.plot(best_interp[0], best_interp[1], label='Real Curve', color='blue')
plt.plot(best_interp[0], best_interp[2], label='Interpolated estimated Curve', color='red')
plt.xlabel('Normalized gait phase')
plt.ylabel('Joint angle (in °)')
plt.grid(True)
plt.savefig("hip_abduction_best_estimation_cp_new.png", dpi=600)
#
plt.figure(5, figsize=(14, 6))
plt.scatter(wrst_keypts[0][0], wrst_keypts[0][1], label='Real Key points', color='blue')
plt.scatter(wrst_keypts[1][0], wrst_keypts[1][1], label='Predicted Key points', color='red')
plt.plot(worst_interp[0], worst_interp[1], label='Real Curve', color='blue')
plt.plot(worst_interp[0], worst_interp[2], label='Interpolated estimated Curve', color='red')
plt.xlabel('Normalized gait phase')
plt.ylabel('Joint angle (in °)')
plt.legend()
plt.grid(True)
plt.savefig("hip_abduction_worst_estimation_cp_new.png", dpi=600)

plt.figure(6, figsize=(14, 6))
plt.scatter(typical_keypts[0][0], typical_keypts[0][1], label='Real Key points', color='blue')
plt.scatter(typical_keypts[1][0], typical_keypts[1][1], label='Predicted Key points', color='red')
plt.plot(typical_interp[0], typical_interp[1], label='Real Curve', color='blue')
plt.plot(typical_interp[0], typical_interp[2], label='Interpolated estimated Curve', color='red')
plt.xlabel('Normalized gait phase')
plt.ylabel('Joint angle (in °)')
plt.legend()
plt.grid(True)
plt.savefig("hip_abduction_typical_estimation_cp_new.png", dpi=600)
plt.show()




