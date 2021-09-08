# Openfaceの出力ファイルのpath
face_feature_path = "/Users/fuyan/Documents/"

# base path
path = "/Users/fuyan/LocalDocs/ml-research/"

# ディレクトリ内のcsvファイル
face_feature_csv = path + "csv"

# load colums
columns_loading = [
    " timestamp", " gaze_angle_x", " gaze_angle_y", " pose_Tx",
    " pose_Ty", " pose_Tz", " pose_Rx",
    " pose_Ry", " pose_Rz", " y_62", " y_66",
]

# set header columns
columns_setting_header = [
    " gaze_angle_x", " gaze_angle_y", "gaze_angle_hypo", " pose_Tx", " pose_Ty",
    " pose_Tz", " pose_Rx", " pose_Ry",
    " pose_Rz", "mouth", "y",
]

# set pre feature header columns
columns_setting_pre_feature_header = [
    " gaze_angle_x", " gaze_angle_y", "gaze_angle_hypo",
    " pose_Tx", " pose_Ty", " pose_Tz", " pose_Rx", " pose_Ry",
    " pose_Rz", "mouth", "y", "y_pre_label"  # 数秒前のラベル
]

# 本来のカラム名と値を合わせる

feature_rolling_colums = [
    "y",
    "y_pre_label",
    # ave
    "ave_gaze_x",
    "ave_gaze_y",
    "ave_gaze_hypo",
    "ave_pose_Tx",
    "ave_pose_Ty",
    "ave_pose_Tz",
    "ave_pose_Rx",
    "ave_pose_Ry",
    "ave_pose_Rz",
    "ave_mouth",
    # std
    "std_gaze_x",
    "std_gaze_y",
    "std_gaze_hypo",
    "std_pose_Tx",
    "std_pose_Ty",
    "std_pose_Tz",
    "std_pose_Rx",
    "std_pose_Ry",
    "std_pose_Rz",
    "std_mouth",
    # max
    "max_gaze_x",
    "max_gaze_y",
    "max_gaze_hypo",
    "max_pose_Tx",
    "max_pose_Ty",
    "max_pose_Tz",
    "max_pose_Rx",
    "max_pose_Ry",
    "max_pose_Rz",
    "max_mouth",
    # min
    "min_gaze_x",
    "min_gaze_y",
    "min_gaze_hypo",
    "min_pose_Tx",
    "min_pose_Ty",
    "min_pose_Tz",
    "min_pose_Rx",
    "min_pose_Ry",
    "min_pose_Rz",
    "min_mouth",
    # med
    "med_gaze_x",
    "med_gaze_y",
    "med_gaze_hypo",
    "med_pose_Tx",
    "med_pose_Ty",
    "med_pose_Tz",
    "med_pose_Rx",
    "med_pose_Ry",
    "med_pose_Rz",
    "med_mouth",
    # skew
    "skew_gaze_x",
    "skew_gaze_y",
    "skew_gaze_hypo",
    "skew_pose_Tx",
    "skew_pose_Ty",
    "skew_pose_Tz",
    "skew_pose_Rx",
    "skew_pose_Ry",
    "skew_pose_Rz",
    "skew_mouth",
    # kurt
    "kurt_gaze_x",
    "kurt_gaze_y",
    "kurt_gaze_hypo",
    "kurt_pose_Tx",
    "kurt_pose_Ty",
    "kurt_pose_Tz",
    "kurt_pose_Rx",
    "kurt_pose_Ry",
    "kurt_pose_Rz",
    "kurt_mouth",
]

# 学習用 カラム
feature_reindex_colums = [
    "y",
    "y_pre_label",
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
    # gaze_hypo
    "ave_gaze_hypo",
    "std_gaze_hypo",
    "max_gaze_hypo",
    "min_gaze_hypo",
    "med_gaze_hypo",
    "skew_gaze_hypo",
    "kurt_gaze_hypo",
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
