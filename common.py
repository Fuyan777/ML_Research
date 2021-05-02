feature_colums = [
    "ave_gaze_x",
    "std_gaze_x",
    "min_gaze_x",
    "max_gaze_x",
    "median_gaze_x",
    "ave_gaze_y",
    "std_gaze_y",
    "min_gaze_y",
    "max_gaze_y",
    "median_gaze_y",
    "ave_poze_x",
    "std_poze_x",
    "min_poze_x",
    "max_poze_x",
    "median_poze_x",
    "ave_poze_y",
    "std_poze_y",
    "min_poze_y",
    "max_poze_y",
    "median_poze_y",
    "ave_poze_z",
    "std_poze_z",
    "min_poze_z",
    "max_poze_z",
    "median_poze_z",
    "ave_mouth",
    "std_mouth",
    "min_mouth",
    "max_mouth",
    "median_mouth",
]

feature_rolling_colums = [
    # ave
    "ave_gaze_x",
    "ave_gaze_y",
    "ave_poze_Tx",
    "ave_poze_Ty",
    "ave_poze_Tz",
    "ave_poze_Rx",
    "ave_poze_Ry",
    "ave_poze_Rz",
    "ave_mouth",
    # std
    "std_gaze_x",
    "std_gaze_y",
    "std_poze_Tx",
    "std_poze_Ty",
    "std_poze_Tz",
    "std_poze_Rx",
    "std_poze_Ry",
    "std_poze_Rz",
    "std_mouth",
    # max
    "max_gaze_x",
    "max_gaze_y",
    "max_poze_Tx",
    "max_poze_Ty",
    "max_poze_Tz",
    "max_poze_Rx",
    "max_poze_Ry",
    "max_poze_Rz",
    "max_mouth",
    # min
    "min_gaze_x",
    "min_gaze_y",
    "min_poze_Tx",
    "min_poze_Ty",
    "min_poze_Tz",
    "min_poze_Rx",
    "min_poze_Ry",
    "min_poze_Rz",
    "min_mouth",
    # med
    "med_gaze_x",
    "med_gaze_y",
    "med_poze_Tx",
    "med_poze_Ty",
    "med_poze_Tz",
    "med_poze_Rx",
    "med_poze_Ry",
    "med_poze_Rz",
    "med_mouth",
    # skew
    "skew_gaze_x",
    "skew_gaze_y",
    "skew_poze_Tx",
    "skew_poze_Ty",
    "skew_poze_Tz",
    "skew_poze_Rx",
    "skew_poze_Ry",
    "skew_poze_Rz",
    "skew_mouth",
    # skew
    "kurt_gaze_x",
    "kurt_gaze_y",
    "kurt_poze_Tx",
    "kurt_poze_Ty",
    "kurt_poze_Tz",
    "kurt_poze_Rx",
    "kurt_poze_Ry",
    "kurt_poze_Rz",
    "kurt_mouth",
]

feature_colums_reindex = [
    # gaze_x
    "ave_gaze_x",
    "std_gaze_x",
    "max_gaze_x",
    "min_gaze_x",
    "med_gaze_x",
    "skew_gaze_x",
    "kurt_gaze_x",
    # gaze_y
    "ave_gaze_y",
    "std_gaze_y",
    "max_gaze_y",
    "min_gaze_y",
    "med_gaze_y",
    "skew_gaze_y",
    "kurt_gaze_y",
    # poze_Tx
    "ave_pose_Tx",
    "std_pose_Tx",
    "max_pose_Tx",
    "min_pose_Tx",
    "med_pose_Tx",
    "skew_pose_Tx",
    "kurt_pose_Tx",
    # poze_Ty
    "ave_pose_Ty",
    "std_pose_Ty",
    "max_pose_Ty",
    "min_pose_Ty",
    "med_pose_Ty",
    "skew_pose_Ty",
    "kurt_pose_Ty",
    # poze_Tz
    "ave_pose_Tz",
    "std_pose_Tz",
    "max_pose_Tz",
    "min_pose_Tz",
    "med_pose_Tz",
    "skew_pose_Tz",
    "kurt_pose_Tz",
    # poze_Rx
    "ave_pose_Rx",
    "std_pose_Rx",
    "max_pose_Rx",
    "min_pose_Rx",
    "med_pose_Rx",
    "skew_pose_Rx",
    "kurt_pose_Rx",
    # poze_Ry
    "ave_pose_Ry",
    "std_pose_Ry",
    "max_pose_Ry",
    "min_pose_Ry",
    "med_pose_Ry",
    "skew_pose_Ry",
    "kurt_pose_Ry",
    # poze_Rz
    "ave_pose_Rz",
    "std_pose_Rz",
    "max_pose_Rz",
    "min_pose_Rz",
    "med_pose_Rz",
    "skew_pose_Rz",
    "kurt_pose_Rz",
    # mouth
    "ave_mouth",
    "std_mouth",
    "max_mouth",
    "min_mouth",
    "med_mouth",
    "skew_mouth",
    "kurt_mouth",
]

speak_columns = [
    "y",
    # gaze_x
    "ave_gaze_x",
    "std_gaze_x",
    "max_gaze_x",
    "min_gaze_x",
    "med_gaze_x",
    "skew_gaze_x",
    "kurt_gaze_x",
    # gaze_y
    "ave_gaze_y",
    "std_gaze_y",
    "max_gaze_y",
    "min_gaze_y",
    "med_gaze_y",
    "skew_gaze_y",
    "kurt_gaze_y",
    # poze_Tx
    "ave_pose_Tx",
    "std_pose_Tx",
    "max_pose_Tx",
    "min_pose_Tx",
    "med_pose_Tx",
    "skew_pose_Tx",
    "kurt_pose_Tx",
    # poze_Ty
    "ave_pose_Ty",
    "std_pose_Ty",
    "max_pose_Ty",
    "min_pose_Ty",
    "med_pose_Ty",
    "skew_pose_Ty",
    "kurt_pose_Ty",
    # poze_Tz
    "ave_pose_Tz",
    "std_pose_Tz",
    "max_pose_Tz",
    "min_pose_Tz",
    "med_pose_Tz",
    "skew_pose_Tz",
    "kurt_pose_Tz",
    # poze_Rx
    "ave_pose_Rx",
    "std_pose_Rx",
    "max_pose_Rx",
    "min_pose_Rx",
    "med_pose_Rx",
    "skew_pose_Rx",
    "kurt_pose_Rx",
    # poze_Ry
    "ave_pose_Ry",
    "std_pose_Ry",
    "max_pose_Ry",
    "min_pose_Ry",
    "med_pose_Ry",
    "skew_pose_Ry",
    "kurt_pose_Ry",
    # poze_Rz
    "ave_pose_Rz",
    "std_pose_Rz",
    "max_pose_Rz",
    "min_pose_Rz",
    "med_pose_Rz",
    "skew_pose_Rz",
    "kurt_pose_Rz",
    # mouth
    "ave_mouth",
    "std_mouth",
    "max_mouth",
    "min_mouth",
    "med_mouth",
    "skew_mouth",
    "kurt_mouth",
]
