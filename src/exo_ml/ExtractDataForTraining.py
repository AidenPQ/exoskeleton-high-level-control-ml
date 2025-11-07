import DatasetsFiltering
from KeyPointsExtractions import KeyPointsExtractions
from CurvesReconstruction import CurveReconstruction
from scipy.interpolate import make_interp_spline
import h5py
import numpy as np
import random

key_points_extraction = KeyPointsExtractions()
curve_reconstruction = CurveReconstruction()


def extract_data_to_train(dataset_file, joint, direction, curve_repetition, attributes_to_include,
                          conditions_on_attributes, real_time_extraction=False):
    data_file = h5py.File(dataset_file, 'r')
    data_file_subjects = list(data_file.keys())
    X_inputs = []
    timesteps = []
    y_trajectories = []
    for subject_id in data_file_subjects:
        subject_data = data_file[subject_id]

        # Extract subject details as input
        subject = []
        subject_details = subject_data["subjectdetails"]

        if all_conditions_met(conditions_on_attributes["Age"], subject_details["Age"][()]) and all_conditions_met(
                conditions_on_attributes["Gender"], subject_details["Gender"][()]):

            for attr in attributes_to_include:
                subject.append(subject_details[attr][()])

            # Extract Speed from experiment
            subject_data_attributes = list(subject_data.keys())
            subject_data_attributes.remove('subjectdetails')

            for experiment in subject_data_attributes:
                if subject_data[experiment]["description"]["Incline"][()] == 0:
                    sides = ["left", "right"]
                    for side in sides:
                        subject_inputs = subject + [subject_data[experiment]["description"]["Speed"][()]]
                        subject_experiments_attr = list(subject_data[experiment][side])
                        if "time" in subject_experiments_attr:
                            time = subject_data[experiment][side]["time"]
                        else:
                            time = subject_data[experiment][side]["time_norm"]
                        jointprofile = subject_data[experiment][side]["angles"][joint][direction]
                        if all(isinstance(elem, np.ndarray) for elem in time):
                            for i in range(len(time)):
                                time_norm = (time[i] / max(time[i])) * 100
                                for j in range(curve_repetition):
                                    pos_choice_index = random.randint(0, (len(time[i]) - 1))
                                    if real_time_extraction:
                                        timesteps.append(time[i])
                                    else:
                                        timesteps.append(time_norm)
                                    y_trajectories.append(jointprofile[i])
                                    x_list = subject_inputs + [time_norm[pos_choice_index],
                                                               jointprofile[i][pos_choice_index]]
                                    X_inputs.append(x_list)
                        else:
                            time_norm = (time / max(time)) * 100
                            for j in range(curve_repetition):
                                pos_choice_index = random.randint(0, (len(time) - 1))
                                timesteps.append(time_norm)
                                x_list = subject_inputs + [time_norm[pos_choice_index], jointprofile[pos_choice_index]]
                                X_inputs.append(x_list)
                                y_trajectories.append(jointprofile)

    return X_inputs, timesteps, y_trajectories


def compute_data_for_training(dataset_file, joint, direction, curve_repetition, attributes_to_include,
                              conditions_on_attributes):
    X_inputs, time_steps, y_trajectories = extract_data_to_train(dataset_file, joint, direction, curve_repetition,
                                                                 attributes_to_include, conditions_on_attributes)

    y_data = []
    for i in range(len(X_inputs)):
        begin_point, begin_point_value, filtered_extremum_pos, extremum_values, filtered_curvature_inversion_pos, curvature_inversion_values, more_points_pos, more_points_values, time_norm = key_points_extraction.key_points_estimation(
            time=time_steps[i], joint_function=y_trajectories[i], joint=joint, direction=direction)

        sorted_keypoints_pos_list, sorted_keypoints_values_list = curve_reconstruction.recompose_keypoints(
            extremum_pos=filtered_extremum_pos,
            extremum_values=extremum_values,
            curvature_inversion_pos=filtered_curvature_inversion_pos,
            curvature_inversion_values=curvature_inversion_values,
            begin_point=begin_point,
            begin_point_value=begin_point_value,
            more_points_pos=more_points_pos,
            more_points_values=more_points_values,
            time_norm=time_norm)
        y = []
        for j in range(len(sorted_keypoints_pos_list)):
            y += [np.float64(sorted_keypoints_pos_list[j] / 100), (np.float64(sorted_keypoints_values_list[j]) * (np.pi / 180))]
            # y += [np.float64(sorted_keypoints_pos_list[j] / 100),
            #       (np.float64(sorted_keypoints_values_list[j]))]
        y_data.append(y)
    return X_inputs, y_data, y_trajectories, time_steps


def compute_data_gait_period_training(dataset_file):
    conditions_on_attributes = {
        "Age": None,
        "Gender": None
    }
    attributes_to_include = ["Age", "Gender", "Height", "Weight"]
    inputs = []
    periods = []
    x_inputs, time_steps, y = extract_data_to_train(dataset_file=dataset_file,
                                                    joint="knee",
                                                    direction="x",
                                                    curve_repetition=1,
                                                    conditions_on_attributes=conditions_on_attributes,
                                                    attributes_to_include=attributes_to_include,
                                                    real_time_extraction=True)

    for i in range(len(x_inputs)):
        period = max(time_steps[i])
        if period <= 10:
            inputs.append(x_inputs[i][:-2])
            periods.append(period)
        else:
            stride_length = 0.65 * (x_inputs[i][2] / 1000)
            gait_period = stride_length / x_inputs[i][4]
            # print("Height:", x_inputs[i][2])
            # print("Speed:", x_inputs[i][4])
            # print("Gait periode:", gait_period)
            inputs.append(x_inputs[i][:-2])
            periods.append(gait_period)

    assert len(inputs) == len(periods)

    return inputs, periods


def compute_data_intent_recognition(dataset_file, gait_percentage_number, transformed=True):
    conditions_on_attributes = {
        "Age": None,
        "Gender": None
    }
    attributes_to_include = ["Age", "Gender", "Height", "Weight"]
    inputs = []
    gait_percentages = []
    x_inputs_knee, time_steps_knee, y_knee = extract_data_to_train(dataset_file=dataset_file, joint="knee", direction="x", curve_repetition=1, conditions_on_attributes=conditions_on_attributes, attributes_to_include=attributes_to_include, real_time_extraction=True)
    x_inputs_hip_x, time_steps_hip_x, y_hip_x = extract_data_to_train(dataset_file=dataset_file, joint="hip", direction="x", curve_repetition=1, conditions_on_attributes=conditions_on_attributes, attributes_to_include=attributes_to_include, real_time_extraction=True)
    x_inputs_hip_y, time_steps_hip_y, y_hip_y = extract_data_to_train(dataset_file=dataset_file, joint="knee", direction="x", curve_repetition=1, conditions_on_attributes=conditions_on_attributes, attributes_to_include=attributes_to_include, real_time_extraction=True)

    assert_lists_equal(time_steps_knee, time_steps_hip_x)
    assert_lists_equal(time_steps_knee, time_steps_hip_y)
    assert len(time_steps_knee) == len(y_hip_y)
    assert len(time_steps_knee) == len(y_hip_x)

    for i in range(len(time_steps_knee)):
        knee_curve = make_interp_spline(time_steps_knee[i], y_knee[i], 3)
        hip_x_curve = make_interp_spline(time_steps_knee[i], y_hip_x[i], 3)
        hip_y_curve = make_interp_spline(time_steps_knee[i], y_hip_y[i], 3)
        knee_curve_derivative = knee_curve.derivative()
        hip_x_curve_derivative = hip_x_curve.derivative()
        hip_y_curve_derivative = hip_y_curve.derivative()

        period = max(time_steps_knee[i])
        if period < 10:
            for j in range(gait_percentage_number):
                gait_index = random.randint(0, len(time_steps_knee[i]) - 1)
                time_step = time_steps_knee[i][gait_index]
                gait_percent = (time_step / period) * 100
                if(transformed):
                    gait_percent_t = (np.pi / 100)*gait_percent - (np.pi/2)
                    gait_percentages.append(gait_percent_t)
                else:
                    gait_percentages.append(gait_percent)
                inputs.append([
                    knee_curve(time_step),
                    knee_curve_derivative(time_step),
                    hip_x_curve(time_step),
                    hip_x_curve_derivative(time_step),
                    hip_y_curve(time_step),
                    hip_y_curve_derivative(time_step)
                ])

    return inputs, gait_percentages


def assert_lists_equal(l1, l2):
    assert len(l1) == len(l2), f"The lists are of different lengths: {len(l1)} != {len(l2)}"
    for a, b in zip(l1, l2):
        if isinstance(a, list) and isinstance(b, list):
            assert_lists_equal(a, b)
        elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            assert np.array_equal(a, b), f"Arrays are different:\n{a}\n!=\n{b}"
        else:
            assert a == b, f"Elements {a} and {b} are different"


def recompose_output(y_output):
    key_points_pos = []
    key_points_values = []
    for i in range(0, len(y_output), 2):
        key_points_pos.append(y_output[i])
        key_points_values.append(y_output[i + 1])
    return key_points_pos, key_points_values


def all_conditions_met(conditions, x):
    if conditions is None:
        return True
    else:
        for condition in conditions:
            if condition is not None and not condition(x):
                return False
        return True

