import h5py
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
import random


class KeyPointsExtractions:
    # def extract_data_from_file(self, dataset_file, joint, side, direction, curve_repetition):
    #     data_file = h5py.File(dataset_file, 'r')
    #     data_file_subjects = list(data_file.keys())
    #     X_inputs = []
    #     timesteps = []
    #     y_trajectories = []
    #     for subject_id in data_file_subjects:
    #         subject_data = data_file[subject_id]
    #
    #         # Extract subject details as input
    #         subject = []
    #         subject_details = subject_data["subjectdetails"]
    #         subject_details_attributes = list(subject_details.keys())
    #
    #         for attr in subject_details_attributes:
    #             subject.append(subject_details[attr][()])
    #
    #         # Extract Speed from experiment
    #         subject_data_attributes = list(subject_data.keys())
    #         subject_data_attributes.remove('subjectdetails')
    #
    #         for experiment in subject_data_attributes:
    #             if subject_data[experiment]["description"]["Incline"][()] == 0:
    #                 subject_inputs = subject + [subject_data[experiment]["description"]["Speed"][()]]
    #                 subject_experiments_attr = list(subject_data[experiment][side])
    #                 if "time" in subject_experiments_attr:
    #                     time = subject_data[experiment][side]["time"]
    #                 else:
    #                     time = subject_data[experiment][side]["time_norm"]
    #                 jointprofile = subject_data[experiment][side]["angles"][joint][direction]
    #                 if all(isinstance(elem, np.ndarray) for elem in time):
    #                     for i in range(len(time)):
    #                         time_norm = (time[i] / max(time[i])) * 100
    #                         for j in range(curve_repetition):
    #                             pos_choice_index = random.randint(0, (len(time[i]) - 1))
    #                             timesteps.append(time_norm)
    #                             y_trajectories.append(jointprofile[i])
    #                             X_inputs.append(subject_inputs + [time_norm[pos_choice_index], jointprofile[i][pos_choice_index]])
    #                 else:
    #                     time_norm = (time / max(time)) * 100
    #                     for j in range(curve_repetition):
    #                         pos_choice_index = random.randint(0, (len(time)-1))
    #                         timesteps.append(time_norm)
    #                         X_inputs.append(subject_inputs + [time_norm[pos_choice_index], jointprofile[pos_choice_index]])
    #                         y_trajectories.append(jointprofile)
    #     return X_inputs, timesteps, y_trajectories

    def hip_flexion_key_points_conditions(self, extremum_pos, curvature_inversion_pos, spline, spline_derivative):

        if len(extremum_pos) > 2:
            # Categorize values based on the given ranges
            first_min_potential = [x for x in extremum_pos if 40 < x < 65]
            first_max_potential = [x for x in extremum_pos if 60 < x < 100]

            first_min_values = spline(first_min_potential)
            first_max_values = spline(first_max_potential)

            select_first_min = first_min_potential[np.argmin(first_min_values)]

            select_first_max = first_max_potential[np.argmax(first_max_values)]

            filtered_extremum_pos = [select_first_min, select_first_max]

        else:
            filtered_extremum_pos = extremum_pos

        if len(curvature_inversion_pos) > 1:
            first_curvature_inversion_potential = [x for x in curvature_inversion_pos if
                                                    filtered_extremum_pos[0] < x < filtered_extremum_pos[1]]

            # Choice first curve inversion
            first_curvature_inversion_values = spline_derivative(first_curvature_inversion_potential)
            select_first_curvature_inv = first_curvature_inversion_potential[
                np.argmin(first_curvature_inversion_values)]

            filtered_curvature_inversion_pos = [select_first_curvature_inv]

        else:
            filtered_curvature_inversion_pos = curvature_inversion_pos

        return filtered_extremum_pos, filtered_curvature_inversion_pos

    def hip_abduction_key_points_conditions(self, extremum_pos, curvature_inversion_pos, spline, spline_derivative):
        if len(extremum_pos) > 2:
            # Categorize values based on the given ranges
            first_max_potential = [x for x in extremum_pos if 0 < x < 30]
            first_min_potential = [x for x in extremum_pos if 55 < x < 80]

            first_max_values = spline(first_max_potential)
            first_min_values = spline(first_min_potential)

            select_first_max = first_max_potential[np.argmax(first_max_values)]
            select_first_min = first_min_potential[np.argmin(first_min_values)]

            filtered_extremum_pos = [select_first_max, select_first_min]

        else:
            filtered_extremum_pos = extremum_pos

        if len(curvature_inversion_pos) > 1:
            first_curvature_inversion_potential = [x for x in curvature_inversion_pos if filtered_extremum_pos[0] <= x < filtered_extremum_pos[1]]

            # Choice first curve inversion
            first_curvature_inversion_values = spline_derivative(first_curvature_inversion_potential)
            select_first_curvature_inv = first_curvature_inversion_potential[np.argmin(first_curvature_inversion_values)]

            filtered_curvature_inversion_pos = [select_first_curvature_inv]

        else:
            filtered_curvature_inversion_pos = curvature_inversion_pos

        return filtered_extremum_pos, filtered_curvature_inversion_pos

    def knee_flexion_key_points_conditions(self, extremum_pos, curvature_inversion_pos, spline, spline_derivative):

        if len(extremum_pos) > 3:
            # Categorize values based on the given ranges
            first_max_potential = [x for x in extremum_pos if 0 <= x < 20]
            first_min_potential = [x for x in extremum_pos if 20 < x < 60]
            second_max_potential = [x for x in extremum_pos if 60 < x < 90]

            first_max_values = spline(first_max_potential)
            first_min_values = spline(first_min_potential)
            second_max_values = spline(second_max_potential)

            select_first_max = first_max_potential[np.argmax(first_max_values)]
            select_first_min = first_min_potential[np.argmin(first_min_values)]
            select_second_max = second_max_potential[np.argmax(second_max_values)]

            filtered_extremum_pos = [select_first_max, select_first_min, select_second_max]

        else:
            filtered_extremum_pos = extremum_pos

        if len(curvature_inversion_pos) > 2:
            first_curvature_inversion_potential = [x for x in curvature_inversion_pos if filtered_extremum_pos[0] < x < filtered_extremum_pos[1]]
            second_curvature_inversion_potential = [x for x in curvature_inversion_pos if
                                                   filtered_extremum_pos[1] < x < filtered_extremum_pos[2]]

            # Choice second curve inversion
            first_curvature_inversion_values = spline_derivative(first_curvature_inversion_potential)
            select_first_curvature_inv = first_curvature_inversion_potential[
                np.argmin(first_curvature_inversion_values)]

            # Choice third curve inversion
            second_curvature_inversion_values = spline_derivative(second_curvature_inversion_potential)
            select_second_curvature_inv = second_curvature_inversion_potential[
                np.argmax(second_curvature_inversion_values)]

            filtered_curvature_inversion_pos = [select_first_curvature_inv, select_second_curvature_inv]

        else:
            filtered_curvature_inversion_pos = curvature_inversion_pos

        return filtered_extremum_pos, filtered_curvature_inversion_pos

    def key_points_estimation(self, time, joint_function, joint, direction):
        period = max(time)
        time_norm = (time / period) * 100

        spline_interpolation = CubicSpline(time_norm, joint_function)
        first_derivative = spline_interpolation.derivative()
        second_derivative = first_derivative.derivative()

        begin_point = time_norm[0]
        begin_point_value = joint_function[0]

        # Extract the extremum and curvature inversion
        extremum_pos = []
        curvature_inversion_pos = []
        for i in range(len(time_norm) - 1):
            a, b = time_norm[i], time_norm[i + 1]
            if first_derivative(a) * first_derivative(b) < 0:
                root_first_derivative = brentq(first_derivative, a, b)
                extremum_pos.append(root_first_derivative)
            if second_derivative(a) * second_derivative(b) < 0:
                root_second_derivative = brentq(second_derivative, a, b)
                curvature_inversion_pos.append(root_second_derivative)

        curvature_inversion_pos.sort()
        extremum_pos.sort()
        if joint == "knee":
            filtered_extremum_pos, filtered_curvature_inversion_pos = self.knee_flexion_key_points_conditions(extremum_pos=extremum_pos, curvature_inversion_pos=curvature_inversion_pos, spline=spline_interpolation, spline_derivative=first_derivative)

            extremum_values = [spline_interpolation(x)[()] for x in filtered_extremum_pos]
            curvature_inversion_values = [spline_interpolation(x)[()] for x in filtered_curvature_inversion_pos]

            more_points_pos = [((begin_point + filtered_extremum_pos[0]) / 2)]

            # Alternativ key points
            for i in range(len(filtered_curvature_inversion_pos)):
                more_points_pos.append(((filtered_extremum_pos[i] + filtered_curvature_inversion_pos[i])/2))

            # for i in range(len(filtered_extremum_pos) - 1):
            #     more_points_pos.append(((filtered_extremum_pos[i] + filtered_extremum_pos[i+1])/2))

            more_points_pos.append(((time_norm[-1] + filtered_extremum_pos[-1]) / 2))
            more_points_pos.append((0.8 * time_norm[-1] + 0.2 * filtered_extremum_pos[-1]))
            more_points_values = [spline_interpolation(x)[()] for x in more_points_pos]

        elif joint == "hip" and direction == "x":
            filtered_extremum_pos, filtered_curvature_inversion_pos = self.hip_flexion_key_points_conditions(
                extremum_pos=extremum_pos, curvature_inversion_pos=curvature_inversion_pos, spline=spline_interpolation,
                spline_derivative=first_derivative)

            extremum_values = [spline_interpolation(x)[()] for x in filtered_extremum_pos]
            curvature_inversion_values = [spline_interpolation(x)[()] for x in filtered_curvature_inversion_pos]

            # more_points_pos = [(0.9 * begin_point + 0.1 * filtered_extremum_pos[0]),
            #                    ((begin_point + filtered_extremum_pos[0]) / 2),
            #                    (0.25 * begin_point + 0.75 * filtered_extremum_pos[0]),
            #                    (0.9 * filtered_extremum_pos[0] + 0.1 * filtered_extremum_pos[1]),
            #                    (0.25 * filtered_extremum_pos[0] + 0.75 * filtered_extremum_pos[1]),
            #                    (0.1 * filtered_extremum_pos[0] + 0.9 * filtered_extremum_pos[1]),
            #                    ((time_norm[-1] + filtered_extremum_pos[-1]) / 2)]

            # Alternativ Key points
            more_points_pos = [(0.9 * begin_point + 0.1 * filtered_extremum_pos[0]),
                               ((begin_point + filtered_extremum_pos[0]) / 2),
                               (0.25 * begin_point + 0.75 * filtered_extremum_pos[0]),
                               (0.9 * filtered_extremum_pos[0] + 0.1 * filtered_curvature_inversion_pos[0]),
                               (0.25 * filtered_curvature_inversion_pos[0] + 0.75 * filtered_extremum_pos[1]),
                               (0.1 * filtered_curvature_inversion_pos[0] + 0.9 * filtered_extremum_pos[1]),
                               ((time_norm[-1] + filtered_extremum_pos[-1]) / 2)]

            more_points_values = [spline_interpolation(x)[()] for x in more_points_pos]

        elif joint == "hip" and direction == "y":
            filtered_extremum_pos, filtered_curvature_inversion_pos = self.hip_abduction_key_points_conditions(
                extremum_pos=extremum_pos, curvature_inversion_pos=curvature_inversion_pos, spline=spline_interpolation,
                spline_derivative=first_derivative)
            extremum_values = [spline_interpolation(x)[()] for x in filtered_extremum_pos]
            curvature_inversion_values = [spline_interpolation(x)[()] for x in filtered_curvature_inversion_pos]
            # more_points_pos = [((begin_point + filtered_extremum_pos[0]) / 2),
            #                    ((filtered_extremum_pos[0] + filtered_extremum_pos[1]) / 2),
            #                    (0.6 * filtered_extremum_pos[0] + 0.4 * filtered_extremum_pos[1]),
            #                    (0.75 * filtered_extremum_pos[0] + 0.25 * filtered_extremum_pos[1]),
            #                    (0.4 * filtered_extremum_pos[0] + 0.6 * filtered_extremum_pos[1]),
            #                    ((time_norm[-1] + filtered_extremum_pos[-1]) / 2),
            #                    (0.8 * time_norm[-1] + 0.2 * filtered_extremum_pos[-1])]

            # Alternativ Key points
            more_points_pos = [((begin_point + filtered_extremum_pos[0]) / 2),
                               ((filtered_extremum_pos[0] + filtered_curvature_inversion_pos[0]) / 2),
                               (0.6 * filtered_extremum_pos[0] + 0.4 * filtered_curvature_inversion_pos[0]),
                               (0.75 * filtered_extremum_pos[0] + 0.25 * filtered_curvature_inversion_pos[0]),
                               (0.4 * filtered_extremum_pos[0] + 0.6 * filtered_curvature_inversion_pos[0]),
                               ((time_norm[-1] + filtered_extremum_pos[-1]) / 2),
                               (0.8 * time_norm[-1] + 0.2 * filtered_extremum_pos[-1])]

            more_points_values = [spline_interpolation(x)[()] for x in more_points_pos]

        return begin_point, begin_point_value, filtered_extremum_pos, extremum_values, filtered_curvature_inversion_pos, curvature_inversion_values, more_points_pos, more_points_values, time_norm

    def save_data_for_training(self, data):
        return 0


