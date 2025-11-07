from KeyPointsExtractions import KeyPointsExtractions
from CurvesReconstruction import CurveReconstruction
from ExtractDataForTraining import compute_data_for_training, extract_data_to_train
import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt

# Change font size globally
plt.rcParams.update({'font.size': 14})


def rmse(real_curve, key_points_interp):
    if len(real_curve) == len(key_points_interp):
        return np.sqrt(np.mean((real_curve - key_points_interp) ** 2))


def list_speed_rmse(filename, attribute, joint, direction, interp_degree, error_type="absolute_root_mean_square_error"):
    data_file = h5py.File(filename, 'r')
    curve_reconstruction = CurveReconstruction()
    key_points_extraction = KeyPointsExtractions()
    subjects_id = list(data_file.keys())
    worst_interpolation = 0
    best_interpolation = 0
    max_error = 0
    min_error = 1
    bad_cases = 0

    # List to return
    attr_list = []
    corresponding_rmse = []

    for subject_id in subjects_id:
        subject_data = data_file[subject_id]

        if attribute != "Speed":
            attr_for_subject = subject_data["subjectdetails"][attribute][()]

        subject_data_attributes = list(subject_data.keys())
        subject_data_attributes.remove('subjectdetails')

        sides = ["left", "right"]

        for experiment_id in subject_data_attributes:
            if attribute == "Speed":
                attr_for_subject = subject_data[experiment_id]["description"][attribute][()]
                if attr_for_subject == 0:
                    print(subject_id)
            for side in sides:
                time = subject_data[experiment_id][side]["time"]
                if all(isinstance(elem, np.ndarray) for elem in time):
                    for i in range(len(time)):
                        joint_profile = subject_data[experiment_id][side]["angles"][joint][direction][i]
                        begin_point, begin_point_value, extremum_pos, extremum_values, curvature_inversion_pos, curvature_inversion_values, more_points_pos, more_points_values, time_norm = key_points_extraction.key_points_estimation(
                            time[i], joint_profile, joint=joint, direction=direction)
                        sorted_pos_list, sorted_values_list = curve_reconstruction.recompose_keypoints(extremum_pos=extremum_pos, extremum_values=extremum_values, curvature_inversion_pos=curvature_inversion_pos, curvature_inversion_values=curvature_inversion_values, begin_point=begin_point, begin_point_value=begin_point_value, more_points_pos=more_points_pos, more_points_values=more_points_values, time_norm=time_norm)
                        spline_interpolation = curve_reconstruction.interpolation_sorted_values(sorted_pos=sorted_pos_list, sorted_values=sorted_values_list, interp_degree=interp_degree)
                        interpolation_trajectory = spline_interpolation(time_norm)
                        error_trajectory_and_interpolation = rmse(joint_profile, interpolation_trajectory)
                        attr_list.append(attr_for_subject)
                        if error_trajectory_and_interpolation < min_error:
                            min_error = error_trajectory_and_interpolation
                            best_interpolation = [time_norm, joint_profile, interpolation_trajectory]
                        if error_trajectory_and_interpolation > max_error and error_trajectory_and_interpolation < 20:
                            max_error = error_trajectory_and_interpolation
                            worst_interpolation = [time_norm, joint_profile, interpolation_trajectory]
                        if error_type == "absolute_root_mean_square_error":
                            corresponding_rmse.append(error_trajectory_and_interpolation)
                        elif error_type == "relative_amplitude_root_mean_square_error":
                            max_traj = max(joint_profile)
                            min_traj = min(joint_profile)
                            amplitude = max_traj - min_traj
                            relative_error = error_trajectory_and_interpolation / amplitude
                            corresponding_rmse.append(relative_error)
                            if relative_error > 0.1:
                                bad_cases += 1
                        elif error_type == "relative_mean_root_mean_square_error":
                            mean_trajectory_value = np.mean(joint_profile)
                            corresponding_rmse.append(error_trajectory_and_interpolation / mean_trajectory_value)
                else:
                    joint_profile = subject_data[experiment_id][side]["angles"][joint][direction][:]
                    begin_point, begin_point_value, extremum_pos, extremum_values, curvature_inversion_pos, curvature_inversion_values, more_points_pos, more_points_values, time_norm = key_points_extraction.key_points_estimation(
                        time, joint_profile, joint=joint, direction=direction)
                    sorted_pos_list, sorted_values_list = curve_reconstruction.recompose_keypoints(
                        extremum_pos=extremum_pos, extremum_values=extremum_values,
                        curvature_inversion_pos=curvature_inversion_pos,
                        curvature_inversion_values=curvature_inversion_values, begin_point=begin_point,
                        begin_point_value=begin_point_value, more_points_pos=more_points_pos,
                        more_points_values=more_points_values, time_norm=time_norm)
                    spline_interpolation = curve_reconstruction.interpolation_sorted_values(sorted_pos=sorted_pos_list,
                                                                                            sorted_values=sorted_values_list,
                                                                                            interp_degree=interp_degree)
                    interpolation_trajectory = spline_interpolation(time_norm)
                    error_trajectory_and_interpolation = rmse(joint_profile, interpolation_trajectory)
                    attr_list.append(attr_for_subject)
                    if error_trajectory_and_interpolation < min_error:
                        min_error = error_trajectory_and_interpolation
                        best_interpolation = [time_norm, joint_profile, interpolation_trajectory]
                    if error_trajectory_and_interpolation > max_error and error_trajectory_and_interpolation < 10:
                        max_error = error_trajectory_and_interpolation
                        worst_interpolation = [time_norm, joint_profile, interpolation_trajectory]

                    if error_type == "absolute_root_mean_square_error":
                        corresponding_rmse.append(error_trajectory_and_interpolation)
                    elif error_type == "relative_amplitude_root_mean_square_error":
                        max_traj = max(joint_profile)
                        min_traj = min(joint_profile)
                        amplitude = max_traj - min_traj
                        relative_error = error_trajectory_and_interpolation / amplitude
                        corresponding_rmse.append(relative_error)
                        if relative_error > 0.1:
                            bad_cases += 1
                    elif error_type == "relative_mean_root_mean_square_error":
                        mean_trajectory_value = np.mean(joint_profile)
                        corresponding_rmse.append(error_trajectory_and_interpolation / mean_trajectory_value)
    if error_type == "relative_amplitude_root_mean_square_error":
        amplitude_best = max(best_interpolation[1]) - min(best_interpolation[1])
        amplitude_worst = max(worst_interpolation[1]) - min(worst_interpolation[1])
        min_error = min_error / amplitude_best
        max_error = max_error / amplitude_worst
    return attr_list, corresponding_rmse, max_error, worst_interpolation, min_error, best_interpolation, bad_cases


processed_filename = "processed_data.h5"

corresponding_speed_knee_x_3, interp_error_knee_flexion_by_speed_3, max_error_knee_x, worst_interpolation_knee_x, min_error_knee_x, best_interpolation_knee_x, bad_cases_knee_x = list_speed_rmse(filename=processed_filename, attribute="Speed", joint="knee", direction="x", interp_degree=3, error_type="relative_amplitude_root_mean_square_error")
corresponding_speed_hip_x_3, interp_error_hip_flexion_by_speed_3, max_error_hip_x, worst_interpolation_hip_x, min_error_hip_x, best_interpolation_hip_x, bad_cases_hip_x = list_speed_rmse(filename=processed_filename, attribute="Speed", joint="hip", direction="x", interp_degree=3, error_type="relative_amplitude_root_mean_square_error")
corresponding_speed_hip_y_3, interp_error_hip_abduction_by_speed_3, max_error_hip_y, worst_interpolation_hip_y, min_error_hip_y, best_interpolation_hip_y, bad_cases_hip_y = list_speed_rmse(filename=processed_filename, attribute="Speed", joint="hip", direction="y", interp_degree=3, error_type="relative_amplitude_root_mean_square_error")

# corresponding_speed_knee_x_5, interp_error_knee_flexion_by_speed_5 = list_speed_rmse(filename=processed_filename, attribute="Speed", joint="knee", direction="x", interp_degree=5, error_type="relative_amplitude_root_mean_square_error")
# corresponding_speed_hip_x_5, interp_error_hip_flexion_by_speed_5 = list_speed_rmse(filename=processed_filename, attribute="Speed", joint="hip", direction="x", interp_degree=5, error_type="relative_amplitude_root_mean_square_error")
# corresponding_speed_hip_y_5, interp_error_hip_abduction_by_speed_5 = list_speed_rmse(filename=processed_filename, attribute="Speed", joint="hip", direction="y", interp_degree=5, error_type="relative_amplitude_root_mean_square_error")
#
# corresponding_speed_knee_x_7, interp_error_knee_flexion_by_speed_7 = list_speed_rmse(filename=processed_filename, attribute="Speed", joint="knee", direction="x", interp_degree=7, error_type="relative_amplitude_root_mean_square_error")
# corresponding_speed_hip_x_7, interp_error_hip_flexion_by_speed_7 = list_speed_rmse(filename=processed_filename, attribute="Speed", joint="hip", direction="x", interp_degree=7, error_type="relative_amplitude_root_mean_square_error")
# corresponding_speed_hip_y_7, interp_error_hip_abduction_by_speed_7 = list_speed_rmse(filename=processed_filename, attribute="Speed", joint="hip", direction="y", interp_degree=7, error_type="relative_amplitude_root_mean_square_error")

print("RMSE by interpolation degree for knee flexion:")
print(f'Interpolation degree 3, Mean RMSE {np.mean(interp_error_knee_flexion_by_speed_3)}, Std {np.std(interp_error_knee_flexion_by_speed_3)}, Min error {min_error_knee_x}, Max error {max_error_knee_x}, Bad cases proportion {bad_cases_knee_x / len(corresponding_speed_knee_x_3)}')
# print(f'\nInterpolation degree 5, Mean RMSE {np.mean(interp_error_knee_flexion_by_speed_5)}, Std {np.std(interp_error_knee_flexion_by_speed_5)}')
# print(f'\nInterpolation degree 7, Mean RMSE {np.mean(interp_error_knee_flexion_by_speed_7)}, Std {np.std(interp_error_knee_flexion_by_speed_7)}')

print("RMSE by interpolation degree for hip flexion:")
print(f'Interpolation degree 3, Mean RMSE {np.mean(interp_error_hip_flexion_by_speed_3)}, Std {np.std(interp_error_hip_flexion_by_speed_3)}, Min error {min_error_hip_x}, Max error {max_error_hip_x}, Bad cases proportion {bad_cases_hip_x / len(corresponding_speed_hip_x_3)}')
# print(f'\nInterpolation degree 5, Mean RMSE {np.mean(interp_error_hip_flexion_by_speed_5)}, Std {np.std(interp_error_hip_flexion_by_speed_5)}')
# print(f'\nInterpolation degree 7, Mean RMSE {np.mean(interp_error_hip_flexion_by_speed_7)}, Std {np.std(interp_error_hip_flexion_by_speed_7)}')

print("RMSE by interpolation degree for hip abduction:")
print(f'Interpolation degree 3, Mean RMSE {np.mean(interp_error_hip_abduction_by_speed_3)}, Std {np.std(interp_error_hip_abduction_by_speed_3)}, Min error {min_error_hip_y}, Max error {max_error_hip_y}, Bad cases proportion {bad_cases_hip_y / len(corresponding_speed_hip_y_3)}')
# print(f'\nInterpolation degree 5, Mean RMSE {np.mean(interp_error_hip_abduction_by_speed_5)}, Std {np.std(interp_error_hip_abduction_by_speed_5)}')
# print(f'\nInterpolation degree 7, Mean RMSE {np.mean(interp_error_hip_abduction_by_speed_7)}, Std {np.std(interp_error_hip_abduction_by_speed_7)}')
# Convert lists to a pandas DataFrame
data_knee_x = pd.DataFrame({'speed': corresponding_speed_knee_x_3, 'interpolation_error': interp_error_knee_flexion_by_speed_3})
data_hip_x = pd.DataFrame({'speed': corresponding_speed_hip_x_3, 'interpolation_error': interp_error_hip_flexion_by_speed_3})
data_hip_y = pd.DataFrame({'speed': corresponding_speed_hip_y_3, 'interpolation_error': interp_error_hip_abduction_by_speed_3})

# Group by abscissas and calculate mean and standard deviation
grouped_data_knee_x = data_knee_x.groupby('speed').agg(
    mean_ordinates=('interpolation_error', 'mean'),
    std_ordinates=('interpolation_error', 'std')
).reset_index()

grouped_data_hip_x = data_hip_x.groupby('speed').agg(
    mean_ordinates=('interpolation_error', 'mean'),
    std_ordinates=('interpolation_error', 'std')
).reset_index()

grouped_data_hip_y = data_hip_y.groupby('speed').agg(
    mean_ordinates=('interpolation_error', 'mean'),
    std_ordinates=('interpolation_error', 'std')
).reset_index()

# Plotting the mean and standard deviation
# plt.figure(1, figsize=(14, 6))
# plt.errorbar(grouped_data_knee_x['speed'], grouped_data_knee_x['mean_ordinates'],
#              yerr=grouped_data_knee_x['std_ordinates'], fmt='o', ecolor='r', capsize=5, label='Mean with Std Dev')
#
# plt.xlabel('Walking Speed in (m/s)')
# plt.ylabel('Relative RMSE')
# plt.legend()
# plt.grid(True)
# plt.savefig("knee_flexion_interpolation_error_relative_to_amplitude.png", dpi=600)
#
# plt.figure(2, figsize=(14, 6))
# plt.errorbar(grouped_data_hip_x['speed'], grouped_data_hip_x['mean_ordinates'],
#              yerr=grouped_data_hip_x['std_ordinates'], fmt='o', ecolor='r', capsize=5, label='Mean with Std Dev')
#
# plt.xlabel('Walking Speed in (m/s)')
# plt.ylabel('Relative RMSE')
# plt.legend()
# plt.grid(True)
# plt.savefig("hip_flexion_interpolation_error_relative_to_amplitude.png", dpi=600)
#
# plt.figure(3, figsize=(14, 6))
# plt.errorbar(grouped_data_hip_y['speed'], grouped_data_hip_y['mean_ordinates'],
#              yerr=grouped_data_hip_y['std_ordinates'], fmt='o', ecolor='r', capsize=5, label='Mean with Std Dev')
#
# plt.xlabel('Walking Speed in (m/s)')
# plt.ylabel('Relative RMSE')
# plt.legend()
# plt.grid(True)
# plt.savefig("hip_abduction_interpolation_error_relative_to_amplitude.png", dpi=600)
#
# plt.figure(4, figsize=(14, 6))
# plt.plot(worst_interpolation_knee_x[0], worst_interpolation_knee_x[1], label='Real Curve', color='blue')
# plt.plot(worst_interpolation_knee_x[0], worst_interpolation_knee_x[2], label='Interpolated Curve', color='red')
# plt.xlabel('Gait percentage (in %)')
# plt.ylabel('Joint angle (in °)')
# plt.legend()
# plt.grid(True)
# plt.savefig("knee_flexion_worst_interpolation.png", dpi=600)
#
# plt.figure(5, figsize=(14, 6))
# plt.plot(best_interpolation_knee_x[0], best_interpolation_knee_x[1], label='Real Curve', color='blue')
# plt.plot(best_interpolation_knee_x[0], best_interpolation_knee_x[2], label='Interpolated Curve', color='red')
# plt.xlabel('Gait percentage (in %)')
# plt.ylabel('Joint angle (in °)')
# plt.legend()
# plt.grid(True)
# plt.savefig("knee_flexion_best_interpolation.png", dpi=600)
#
# plt.figure(6, figsize=(14, 6))
# plt.plot(worst_interpolation_hip_x[0], worst_interpolation_hip_x[1], label='Real Curve', color='blue')
# plt.plot(worst_interpolation_hip_x[0], worst_interpolation_hip_x[2], label='Interpolated Curve', color='red')
# plt.xlabel('Gait percentage (in %)')
# plt.ylabel('Joint angle (in °)')
# plt.legend()
# plt.grid(True)
# plt.savefig("hip_flexion_worst_interpolation.png", dpi=600)
#
# plt.figure(7, figsize=(14, 6))
# plt.plot(best_interpolation_hip_x[0], best_interpolation_hip_x[1], label='Real Curve', color='blue')
# plt.plot(best_interpolation_hip_x[0], best_interpolation_hip_x[2], label='Interpolated Curve', color='red')
# plt.xlabel('Gait percentage (in %)')
# plt.ylabel('Joint angle (in °)')
# plt.legend()
# plt.grid(True)
# plt.savefig("hip_flexion_best_interpolation.png", dpi=600)
#
# plt.figure(8, figsize=(14, 6))
# plt.plot(worst_interpolation_hip_y[0], worst_interpolation_hip_y[1], label='Real Curve', color='blue')
# plt.plot(worst_interpolation_hip_y[0], worst_interpolation_hip_y[2], label='Interpolated Curve', color='red')
# plt.xlabel('Gait percentage (in %)')
# plt.ylabel('Joint angle (in °)')
# plt.legend()
# plt.grid(True)
# plt.savefig("hip_abduction_worst_interpolation.png", dpi=600)
#
# plt.figure(9, figsize=(14, 6))
# plt.plot(best_interpolation_hip_y[0], best_interpolation_hip_y[1], label='Real Curve', color='blue')
# plt.plot(best_interpolation_hip_y[0], best_interpolation_hip_y[2], label='Interpolated Curve', color='red')
# plt.xlabel('Gait percentage (in %)')
# plt.ylabel('Joint angle (in °)')
# plt.legend()
# plt.grid(True)
# plt.savefig("hip_abduction_best_interpolation.png", dpi=600)
# plt.show()
