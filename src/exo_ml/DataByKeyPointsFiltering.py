import numpy as np
import h5py
import math
from DatasetsFiltering import DatasExtraction
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq


def extremum_estimation(time, joint_function):
    period = max(time)
    time_norm = (time / period) * 100

    spline_interpolation = CubicSpline(time_norm, joint_function)
    first_derivative = spline_interpolation.derivative()

    # Extract the extremum
    extremum_pos = []
    for i in range(len(time_norm) - 1):
        a, b = time_norm[i], time_norm[i + 1]
        if first_derivative(a) * first_derivative(b) < 0:
            root_first_derivative = brentq(first_derivative, a, b)
            extremum_pos.append(root_first_derivative)
    return extremum_pos, spline_interpolation(extremum_pos)


def number_of_possible_extremum(time, joint_function, joint, direction):
    extremum, values = extremum_estimation(time, joint_function)

    if joint == "knee":
        first_max_potential = [x for x in extremum if 0 <= x < 20]
        first_min_potential = [x for x in extremum if 20 < x < 60]
        second_max_potential = [x for x in extremum if 60 < x < 90]

        number_of_potential_extremum = [len(first_max_potential), len(first_min_potential), len(second_max_potential)]

    elif joint == "hip" and direction == "x":
        first_min_potential = [x for x in extremum if 40 < x < 60]
        first_max_potential = [x for x in extremum if 60 < x < 100]

        number_of_potential_extremum = [len(first_min_potential), len(first_max_potential)]

    elif joint == "hip" and direction == "y":
        first_max_potential = [x for x in extremum if 0 < x < 30]
        first_min_potential = [x for x in extremum if 55 < x < 80]

        number_of_potential_extremum = [len(first_max_potential), len(first_min_potential)]

    return number_of_potential_extremum


def all_non_zero(lst):
    return all(x != 0 for x in lst)


# Function to check if all values have a non-zero length
def all_values_non_empty(dictionary):
    for key, value in dictionary.items():
        if len(value) == 0:
            return False
    return True


# Function to check if all values are finite
def all_values_finite(lst):
    return all(math.isfinite(value) for value in lst)


def data_sorting(filename):
    data_file = h5py.File(filename, 'r')
    data_file_subjects = list(data_file.keys())
    processed_data = []

    filtered_subjects = []
    for subject_id in data_file_subjects:
        subject_data = data_file[subject_id]

        # Extract subject details as input
        subject = []
        subject_details = subject_data["subjectdetails"]

        new_subject_details = {}

        subject_details_attributes = list(subject_details.keys())

        for attr in subject_details_attributes:
            new_subject_details[attr] = subject_details[attr][()]

        subject.append(new_subject_details)

        # Extract Speed from experiment
        subject_data_attributes = list(subject_data.keys())
        subject_data_attributes.remove('subjectdetails')

        sides = ["left", "right"]
        joints = ["knee", "hip"]
        directions = {"knee": ["x"], "hip": ["x", "y"]}

        for experiment in subject_data_attributes:
            experiments_results = {"experiment_id": experiment, "description": {"Speed": 0, "Incline": 0},
                                      "left": {"time": [],
                                               "angles": {"hip": {"x": [], "y": []}, "knee": {"x": []}}},
                                      "right": {"time": [],
                                                "angles": {"hip": {"x": [], "y": []}, "knee": {"x": []}}}
                                      }
            if subject_data[experiment]["description"]["Incline"][()] == 0:
                experiments_results["experiment_id"] = experiment
                experiments_results["description"]["Speed"] = subject_data[experiment]["description"]["Speed"][()]
                for side in sides:
                    subject_experiments_attr = list(subject_data[experiment][side])
                    if "time" in subject_experiments_attr:
                        time = subject_data[experiment][side]["time"]
                    else:
                        time = subject_data[experiment][side]["time_norm"]
                    if all(isinstance(elem, np.ndarray) for elem in time):
                        for i in range(len(time)):
                            selection_criteria = []
                            for joint in joints:
                                for direction in directions[joint]:
                                    jointprofile = subject_data[experiment][side]["angles"][joint][direction][i]
                                    selection_criteria.append(all_non_zero(number_of_possible_extremum(time=time[i], joint_function=jointprofile, joint=joint, direction=direction)))
                            if all(selection_criteria):
                                experiments_results[side]["time"].append(time[i])
                                experiments_results[side]["angles"]["hip"]["x"].append(subject_data[experiment][side]["angles"]["hip"]["x"][i])
                                experiments_results[side]["angles"]["hip"]["y"].append(
                                    subject_data[experiment][side]["angles"]["hip"]["y"][i])
                                experiments_results[side]["angles"]["knee"]["x"].append(
                                    subject_data[experiment][side]["angles"]["knee"]["x"][i])

                    else:
                        selection_criteria = []
                        for joint in joints:
                            for direction in directions[joint]:
                                jointprofile = subject_data[experiment][side]["angles"][joint][direction][:]
                                if not all_values_finite(jointprofile):
                                    jointprofile = [0] * len(time)
                                selection_criteria.append(all_non_zero(
                                    number_of_possible_extremum(time=time, joint_function=jointprofile,
                                                                joint=joint, direction=direction)))
                        if all(selection_criteria):
                            experiments_results[side]["time"] = time
                            experiments_results[side]["angles"]["hip"]["x"] = subject_data[experiment][side]["angles"]["hip"]["x"]
                            experiments_results[side]["angles"]["hip"]["y"] = subject_data[experiment][side]["angles"]["hip"]["y"]
                            experiments_results[side]["angles"]["knee"]["x"] = subject_data[experiment][side]["angles"]["knee"]["x"]

                if len(experiments_results["left"]["time"]) != 0 and len(experiments_results["right"]["time"]) != 0:
                    subject.append(experiments_results)

        if len(subject) > 1:
            processed_data.append(subject)
    return processed_data


# filename_ = "datasetV2.h5"
# processed_file = "processed_data.h5"
# data_extraction = DatasExtraction()
# data = data_sorting(filename_)
# data_extraction.save_data_in_hdf5_file(data=data, filename_data=processed_file, mode='w')
# print(data[0])
