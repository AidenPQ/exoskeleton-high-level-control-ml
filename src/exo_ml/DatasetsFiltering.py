import h5py
import numpy as np
import pandas as pd
from pymatreader import read_mat
import os

# K embry dataset
filename_inclineExperiment = "Datasets/Sources/InclineExperiment/InclineExperiment.mat"

# Fukuchi dataset
filename_subjectdetails_fukuchi = "Datasets/Sources/WBDSinfo.xlsx"
directory_path_trials_fukuchi = "Datasets/Sources/WBDSascii"
directory_path_monteira = "Datasets/Sources/MAT files"

filename_dataset = "dataset.h5"


class DatasExtraction:

    def write_subject_details(self, id, age, gender, height, weight):
        return {"Id": id, "Age": age, "Gender" : gender, "Height": height, "Weight": weight}

    def kembry_file_subdataset_info_extraction(self, data_file, dataset_to_browse, dict_to_complete, dict_key_to_index):
        col_index = 0
        keys = list(dict_key_to_index.keys())
        for col in dataset_to_browse:
            line_index = 0
            for line_object_reference in col:
                if col_index == 1:
                    line_value = data_file[line_object_reference]  # This is how with the object reference we have access to the value which are also stored in table with a vector shape
                    detail_value = line_value[:, 0]
                    for key in keys:
                        if line_index == dict_key_to_index[key]:
                            dict_to_complete[key] = detail_value[0]
                line_index += 1
            col_index += 1

    def k_embry_data_extraction(self):
        Data = []

        data_file = h5py.File(filename_inclineExperiment, "r")
        dataset = data_file["Gaitcycle"]
        subjects = list(dataset.keys())
        for subject in subjects:
            data_sub_set = dataset[subject]
            subset_details_experiment_keys = list(data_sub_set.keys())

            subject_data = []

            # Retrieve details of test subjects

            subject_details = {"Id": subject, "Age": 0, "Gender": 0, "Height": 0, "Weight": 0}
            dict_key_to_index_subjectdetails = {"Gender": 0, "Age": 1, "Height": 2, "Weight": 3}
            self.kembry_file_subdataset_info_extraction(data_file=data_file, dataset_to_browse=data_sub_set["subjectdetails"], dict_to_complete=subject_details, dict_key_to_index=dict_key_to_index_subjectdetails)

            subject_data.append(subject_details)

            for experiment_key in subset_details_experiment_keys:
                sides = ["left", "right"]
                experiment_results = {"experiment_id": experiment_key, "description": {"Speed": 0, "Incline": 0}, "left": {"time": [], "torque": {"hip": {"x": [], "y": []}, "knee": {"x": []}}, "angles": {"hip": {"x": [], "y": []}, "knee": {"x": []}}}, "right": {"time": [], "torque": {"hip": {"x": [], "y": []}, "knee": {"x": []}}, "angles": {"hip": {"x": [], "y": []}, "knee": {"x": []}}}}
                if experiment_key != "subjectdetails":
                    experiment = data_sub_set[experiment_key]

                    # Experiment description extraction
                    dict_key_to_index = {"Speed": 0, "Incline": 1}
                    self.kembry_file_subdataset_info_extraction(data_file=data_file, dataset_to_browse=experiment["description"], dict_to_complete=experiment_results["description"], dict_key_to_index=dict_key_to_index)

                    # Experiment time frame extraction
                    timesteps = experiment["cycles"]
                    for side in sides:
                        timesteps_one_side = timesteps[side]["time"]
                        for each_timesteps_one_side_column in timesteps_one_side:
                            experiment_results[side]["time"].append(each_timesteps_one_side_column)

                    # Experiment joint angles extraction
                    joint_directions = ["x", "y", "x"]
                    joint_list = ["hip", "hip", "knee"]
                    jointangles = experiment["kinematics"]["jointangles"]
                    jointmoment = experiment["kinetics"]["jointmoment"]
                    for side in sides:
                        jointangles_one_side = jointangles[side]
                        jointmoment_one_side = jointmoment[side]
                        # Hip joint angles extraction
                        for index_joint in range(len(joint_list)):
                            joint_angles_one_side = jointangles_one_side[joint_list[index_joint]]
                            joint_moment_one_side = jointmoment_one_side[joint_list[index_joint]]
                            joint_angles_one_side_one_direction = joint_angles_one_side[joint_directions[index_joint]]
                            joint_moment_one_side_one_direction = joint_moment_one_side[joint_directions[index_joint]]
                            for joint_angles_one_side_one_direction_column in joint_angles_one_side_one_direction:
                                experiment_results[side]["angles"][joint_list[index_joint]][joint_directions[index_joint]].append(
                                    joint_angles_one_side_one_direction_column)
                            for joint_moment_one_side_one_direction_column in joint_moment_one_side_one_direction:
                                experiment_results[side]["torque"][joint_list[index_joint]][joint_directions[index_joint]].append(
                                    joint_moment_one_side_one_direction_column)
                    subject_data.append(experiment_results)
            Data.append(subject_data)
        return Data

    def fukuchi_data_extraction(self):
        Data = []
        excel_subject_experiments_details = pd.read_excel(filename_subjectdetails_fukuchi)

        for i in range(1, 43):
            subject_data = []
            # Extract The rows of the file for one subject
            condition = (excel_subject_experiments_details['Subject'] == i) & (excel_subject_experiments_details['FileName'].str.endswith('ang.txt'))

            filtered_rows = excel_subject_experiments_details[condition]

            # Extract subject details
            first_row = filtered_rows.iloc[0]
            if i < 10:
                Id = "WBDS0" + str(i)
            else:
                Id = "WBDS" + str(i)
            subject_details = {"Id": Id, "Age": first_row["Age"], "Gender": 0, "Height": first_row["Height"] * 10, "Weight": first_row["Mass"]}
            if first_row["Gender"] == "M":
                subject_details["Gender"] = 1
            else:
                subject_details["Gender"] = 2

            subject_data.append(subject_details)
            # Extract the experiment
            for j in range(0, len(filtered_rows)):
                filename_measures = filtered_rows.iloc[j]["FileName"]
                if i == 7:
                    filename_measures = filename_measures.replace("WBDS03", Id)
                experiment_key = filename_measures.replace("ang.txt", "")
                experiment_results = {"experiment_id": experiment_key, "description": {"Speed": 0, "Incline": 0},
                                      "left": {"time_norm": [], "torque": {"hip": {"x": [], "y": []}, "knee": {"x": []}}, "angles": {"hip": {"x": [], "y": []}, "knee": {"x": []}}},
                                      "right": {"time_norm": [], "torque": {"hip": {"x": [], "y": []}, "knee": {"x": []}}, "angles": {"hip": {"x": [], "y": []}, "knee": {"x": []}}}}

                # Extract experiment description
                file_identifier_t = Id + "walkT"
                file_identifier_o = Id + "walkO"
                if filename_measures.startswith(file_identifier_t):
                    filename_same_gait_description = experiment_key + "mkr.txt"
                    condition_find_mkr = (excel_subject_experiments_details['Subject'] == i) & (excel_subject_experiments_details['FileName'] == filename_same_gait_description)
                    filtered_rows_find_mkr = excel_subject_experiments_details[condition_find_mkr]
                    speed = filtered_rows_find_mkr["GaitSpeed(m/s)"].tolist()
                    experiment_results["description"]["Speed"] = speed[0]
                elif filename_measures.startswith(file_identifier_o):
                    walking_mode_o = experiment_key.replace(file_identifier_o, "") + "mkr.txt"
                    condition_all_overground_measurements = (excel_subject_experiments_details['Subject'] == i) & (excel_subject_experiments_details['FileName'].str.startswith(file_identifier_o)) & (excel_subject_experiments_details['FileName'].str.endswith(walking_mode_o))
                    filtered_rows_overground_measurements = excel_subject_experiments_details[condition_all_overground_measurements]
                    gaitspeed_column = filtered_rows_overground_measurements["GaitSpeed(m/s)"]
                    experiment_results["description"]["Speed"] = gaitspeed_column.mean()

                # extract joint angles from data
                filename_measures_path = directory_path_trials_fukuchi + "/" + filename_measures
                measures_table = pd.read_csv(filename_measures_path, delimiter="\t")
                experiment_results["left"]["time_norm"] = measures_table["Time"].tolist()
                experiment_results["left"]["angles"]["hip"]["x"] = measures_table["LHipAngleZ"].tolist()
                experiment_results["left"]["angles"]["hip"]["y"] = measures_table["LHipAngleX"].tolist()
                experiment_results["left"]["angles"]["knee"]["x"] = measures_table["LKneeAngleZ"].tolist()

                experiment_results["right"]["time_norm"] = measures_table["Time"].tolist()
                experiment_results["right"]["angles"]["hip"]["x"] = measures_table["RHipAngleZ"].tolist()
                experiment_results["right"]["angles"]["hip"]["y"] = measures_table["RHipAngleX"].tolist()
                experiment_results["right"]["angles"]["knee"]["x"] = measures_table["RKneeAngleZ"].tolist()

                # Extract joint Moments from Data
                filename_moment_measures = filename_measures.replace("ang.txt", "knt.txt")
                filename_moment_measures_path = directory_path_trials_fukuchi + "/" + filename_moment_measures
                moment_measures_table = pd.read_csv(filename_moment_measures_path, delimiter="\t")
                experiment_results["left"]["time_norm"] = moment_measures_table["Time"].tolist()
                experiment_results["left"]["torque"]["hip"]["x"] = moment_measures_table["LHipMomentZ"].tolist()
                experiment_results["left"]["torque"]["hip"]["y"] = moment_measures_table["LHipMomentX"].tolist()
                experiment_results["left"]["torque"]["knee"]["x"] = moment_measures_table["LKneeMomentZ"].tolist()

                experiment_results["right"]["time_norm"] = moment_measures_table["Time"].tolist()
                experiment_results["right"]["torque"]["hip"]["x"] = moment_measures_table["RHipMomentZ"].tolist()
                experiment_results["right"]["torque"]["hip"]["y"] = moment_measures_table["RHipMomentX"].tolist()
                experiment_results["right"]["torque"]["knee"]["x"] = moment_measures_table["RKneeMomentZ"].tolist()
                subject_data.append(experiment_results)

            Data.append(subject_data)
        return Data

    def moreira_data_extraction(self):
        Data = []
        for participant in os.listdir(directory_path_monteira):
            path_to_one_participant = os.path.join(directory_path_monteira, participant)
            participant_description_experiments = []
            for file in os.listdir(path_to_one_participant):
                subject_details = {"Id": participant, "Age": 0, "Gender": 0, "Height": 0, "Weight": 0}
                file_path = os.path.join(path_to_one_participant, file)
                if file.endswith('.txt'):
                    file_doc = open(file_path, 'r')
                    for line in file_doc:
                        words = line.split()
                        if words[0] == 'Age:':
                            subject_details["Age"] = int(words[1])
                        elif words[0] == 'Gender':
                            if words[1] == 'Male':
                                subject_details["Gender"] = 1
                            else:
                                subject_details["Gender"] = 2
                        elif words[0] == 'Body' and words[1] == 'Height:':
                            subject_details['Height'] = float(words[2]) * 100
                        elif words[0] == 'Body' and words[1] == 'Mass:':
                            subject_details['Weight'] = float(words[2])
                    participant_description_experiments.append(subject_details)

                elif file == "Subject.mat":
                    subject = participant.replace("Participant", "Subject")
                    experiment_file = read_mat(file_path)
                    experiment_file_subject = experiment_file[subject]
                    experiment_list = list(experiment_file[subject].keys())
                    sides = {"L": "left", "R": "right"}
                    for experiment in experiment_list:
                        experiment_results = {"experiment_id": experiment, "description": {"Speed": 0, "Incline": 0},
                                              "left": {"time": [],
                                                       "torque": {"hip": {"x": [], "y": []}, "knee": {"x": []}},
                                                       "angles": {"hip": {"x": [], "y": []}, "knee": {"x": []}}},
                                              "right": {"time": [],
                                                        "torque": {"hip": {"x": [], "y": []}, "knee": {"x": []}},
                                                        "angles": {"hip": {"x": [], "y": []}, "knee": {"x": []}}},
                                              }
                        experiment_results["description"]["Speed"] = experiment_file_subject[experiment]["speed"]
                        listofTrials = list(experiment_file_subject[experiment]["L"]["Angles"].keys())

                        for trial in listofTrials:
                            for side in list(sides.keys()):
                                stride_Time = experiment_file_subject[experiment][side]["Angles"][trial]["Stride_Time"]
                                time = np.linspace(0, stride_Time, 1001)
                                experiment_results[sides[side]]["time"].append(time)
                                experiment_results[sides[side]]["angles"]["hip"]["x"].append(experiment_file_subject[experiment][side]["Angles"][trial]["hip"]["x"])
                                experiment_results[sides[side]]["angles"]["hip"]["y"].append(
                                    experiment_file_subject[experiment][side]["Angles"][trial]["hip"]["y"])
                                experiment_results[sides[side]]["angles"]["knee"]["x"].append(
                                    experiment_file_subject[experiment][side]["Angles"][trial]["knee"]["x"])

                                experiment_results[sides[side]]["torque"]["hip"]["x"].append(
                                    experiment_file_subject[experiment][side]["Torques"][trial]["hip"]["x"])
                                experiment_results[sides[side]]["torque"]["hip"]["y"].append(
                                    experiment_file_subject[experiment][side]["Torques"][trial]["hip"]["y"])
                                experiment_results[sides[side]]["torque"]["knee"]["x"].append(
                                    experiment_file_subject[experiment][side]["Torques"][trial]["knee"]["x"])

                        participant_description_experiments.append(experiment_results)

            Data.append(participant_description_experiments)
        return Data

    def save_data_in_hdf5_file(self, data, filename_data, mode='w'):
        sides = ["left", "right"]
        with h5py.File(filename_data, mode) as file:
            for subject_data in data:

                # Create group for each subject
                subject_group = file.create_group(subject_data[0]["Id"])

                # Store the subject details:
                subject_details_group = subject_group.create_group("subjectdetails")
                for key, value in subject_data[0].items():
                    if key != "id":
                        subject_details_group[key] = value

                # Store experiments
                for i in range(1, len(subject_data)):
                    experiment = subject_data[i]

                    # Create group for experiment
                    experiment_group = subject_group.create_group(experiment["experiment_id"])

                    # Store experiment description
                    experiment_description_group = experiment_group.create_group("description")
                    for key, value in experiment["description"].items():
                        experiment_description_group[key] = value

                    # Store joint angles from left and right
                    for side in sides:

                        experiment_side = experiment[side]

                        # Create group for side
                        experiment_side_group = experiment_group.create_group(side)

                        # Store time frame
                        experiment_attributes = list(experiment_side.keys())
                        if "time" in experiment_attributes:
                            time_np = np.array(experiment_side["time"])
                            experiment_side_group.create_dataset("time", data=time_np)
                        else:
                            time_np = np.array(experiment_side["time_norm"])
                            experiment_side_group.create_dataset("time_norm", data=time_np)

                        # Create group for joint angles
                        experiment_side_jointangles_group = experiment_side_group.create_group("angles")

                        # Store hip joint angles
                        experiment_side_jointangles_hip_group = experiment_side_jointangles_group.create_group("hip")
                        angles_hip_x = np.array(experiment_side["angles"]["hip"]["x"])
                        experiment_side_jointangles_hip_group.create_dataset("x", data=angles_hip_x)

                        angles_hip_y = np.array(experiment_side["angles"]["hip"]["y"])
                        experiment_side_jointangles_hip_group.create_dataset("y", data=angles_hip_y)

                        # Store knee joint angles
                        experiment_side_jointangles_knee_group = experiment_side_jointangles_group.create_group("knee")
                        angles_knee_x = np.array(experiment_side["angles"]["knee"]["x"])
                        experiment_side_jointangles_knee_group.create_dataset("x", data=angles_knee_x)

                        experiment_side_keys = list(experiment_side.keys())
                        if "torques" in experiment_side_keys:
                            # Create torque group
                            experiment_side_torque_group = experiment_side_group.create_group("torque")

                            # Store hip joint angles
                            experiment_side_torque_hip_group = experiment_side_torque_group.create_group("hip")
                            torque_hip_x = np.array(experiment_side["torque"]["hip"]["x"])
                            experiment_side_torque_hip_group.create_dataset("x", data=torque_hip_x)

                            torque_hip_y = np.array(experiment_side["torque"]["hip"]["y"])
                            experiment_side_torque_hip_group.create_dataset("y", data=torque_hip_y)

                            # Store knee joint angles
                            experiment_side_torque_knee_group = experiment_side_torque_group.create_group("knee")
                            torque_knee_x = np.array(experiment_side["torque"]["knee"]["x"])
                            experiment_side_torque_knee_group.create_dataset("x", data=torque_knee_x)




