# Openfaceの出力ファイルのpath
face_feature_path = "/Volumes/mac-ssd/face_csv/"

# base path
path = "/Users/fuyan/LocalDocs/ml-research/"

# ディレクトリ内のcsvファイル
face_feature_csv = path + "csv"


# all feature colums
columns_loading_all_feature = [
    " timestamp", " gaze_angle_x", " gaze_angle_y", " pose_Tx",
    " pose_Ty", " pose_Tz", " pose_Rx",
    " pose_Ry", " pose_Rz", " y_62", " y_66",
    " AU01_r", " AU02_r", " AU04_r", " AU05_r",
    " AU06_r", " AU07_r", " AU09_r", " AU10_r", " AU12_r", " AU14_r",
    " AU15_r", " AU17_r", " AU20_r", " AU23_r", " AU25_r", " AU26_r", " AU45_r",
    "isSpeak_other1", "isSpeak_other2"
]

columns_setting_header_all_feature = [
    " gaze_angle_x", " gaze_angle_y", "gaze_angle_hypo", " pose_Tx", " pose_Ty",
    " pose_Tz", " pose_Rx", " pose_Ry", " pose_Rz", "mouth",
    " AU01_r", " AU02_r", " AU04_r", " AU05_r", " AU06_r",
    " AU07_r", " AU09_r", " AU10_r", " AU12_r", " AU14_r", " AU15_r",
    " AU17_r", " AU20_r", " AU23_r", " AU25_r", " AU26_r", " AU45_r", "y"
]

columns_setting_pre_feature_header_all_feature = [
    " gaze_angle_x", " gaze_angle_y", "gaze_angle_hypo",
    " pose_Tx", " pose_Ty", " pose_Tz", " pose_Rx", " pose_Ry",
    " pose_Rz", "mouth",
    " AU01_r", " AU02_r", " AU04_r", " AU05_r", " AU06_r",
    " AU07_r", " AU09_r", " AU10_r", " AU12_r", " AU14_r", " AU15_r",
    " AU17_r", " AU20_r", " AU23_r", " AU25_r", " AU26_r", " AU45_r",
    "y", "y_pre_label", "y_pre_label_0.5s", "y_pre_label_2s", "y_pre_label_3s", "y_pre_label_5s",
]

columns_setting_pre_feature_header_all_feature_re = [
    " gaze_angle_x", " gaze_angle_y", "gaze_angle_hypo",
    " pose_Tx", " pose_Ty", " pose_Tz", " pose_Rx", " pose_Ry",
    " pose_Rz", "mouth",
    " AU01_r", " AU02_r", " AU04_r", " AU05_r", " AU06_r",
    " AU07_r", " AU09_r", " AU10_r", " AU12_r", " AU14_r", " AU15_r",
    " AU17_r", " AU20_r", " AU23_r", " AU25_r", " AU26_r", " AU45_r",
    "y", "y_pre_label", "y_pre_label_0.5s",  "y_pre_label_2s", "y_pre_label_3s", "y_pre_label_5s",
    "isSpeak_other", "isSpeak_other1", "isSpeak_other2", "duration_of_speak_all", "duration_of_speak1", "duration_of_speak2"
]

# index変換用 カラム
feature_all_reindex_colums = [
    "y",
    "y_pre_label", "y_pre_label_0.5s",  "y_pre_label_2s", "y_pre_label_3s", "y_pre_label_5s",
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
    # AU
    "AU01_r_std", "AU01_r_max", "AU01_r_min", "AU01_r_median",  "AU01_r_p25",  "AU01_r_p75",
    "AU02_r_std", "AU02_r_max", "AU02_r_min", "AU02_r_median",  "AU02_r_p25",  "AU02_r_p75",
    "AU04_r_std", "AU04_r_max", "AU04_r_min", "AU04_r_median",  "AU04_r_p25",  "AU04_r_p75",
    "AU05_r_std", "AU05_r_max", "AU05_r_min", "AU05_r_median",  "AU05_r_p25",  "AU05_r_p75",
    "AU06_r_std", "AU06_r_max", "AU06_r_min", "AU06_r_median",  "AU06_r_p25",  "AU06_r_p75",
    "AU07_r_std", "AU07_r_max", "AU07_r_min", "AU07_r_median",  "AU07_r_p25",  "AU07_r_p75",
    "AU09_r_std", "AU09_r_max", "AU09_r_min", "AU09_r_median",  "AU09_r_p25",  "AU09_r_p75",
    "AU10_r_std", "AU10_r_max", "AU10_r_min", "AU10_r_median",  "AU10_r_p25",  "AU10_r_p75",
    "AU12_r_std", "AU12_r_max", "AU12_r_min", "AU12_r_median",  "AU12_r_p25",  "AU12_r_p75",
    "AU14_r_std", "AU14_r_max", "AU14_r_min", "AU14_r_median",  "AU14_r_p25",  "AU14_r_p75",
    "AU15_r_std", "AU15_r_max", "AU15_r_min", "AU15_r_median",  "AU15_r_p25",  "AU15_r_p75",
    "AU17_r_std", "AU17_r_max", "AU17_r_min", "AU17_r_median",  "AU17_r_p25",  "AU17_r_p75",
    "AU20_r_std", "AU20_r_max", "AU20_r_min", "AU20_r_median",  "AU20_r_p25",  "AU20_r_p75",
    "AU23_r_std", "AU23_r_max", "AU23_r_min", "AU23_r_median",  "AU23_r_p25",  "AU23_r_p75",
    "AU25_r_std", "AU25_r_max", "AU25_r_min", "AU25_r_median",  "AU25_r_p25",  "AU25_r_p75",
    "AU26_r_std", "AU26_r_max", "AU26_r_min", "AU26_r_median",  "AU26_r_p25",  "AU26_r_p75",
    "AU45_r_std", "AU45_r_max", "AU45_r_min", "AU45_r_median",  "AU45_r_p25",  "AU45_r_p75",
    # other_speak_status
    "isSpeak_other",
    # "isSpeak_other1",
    # "nonSpeak_other1", "startSpeak_other1", "speaking_other1", "endSpeak_other1",
    # "isSpeak_other2",
    # "nonSpeak_other2", "startSpeak_other2", "speaking_other2", "endSpeak_other2"
]

x_variable_feature_all_colums__ = [
    "max_mouth",
    "AU25_r_std",
    "AU25_r_max",
    "std_mouth",
    "min_mouth",
    "AU06_r_max",
    "AU06_r_min",
    "AU07_r_max",
    "AU04_r_max",
    "AU17_r_min",
    "AU09_r_max",
    "AU09_r_min",
    "AU10_r_max",
    "min_pose_Rx",
    "std_pose_Tx",
    "min_gaze_x",
    "min_pose_Ry",
    "ave_pose_Ry",
    "max_pose_Tz",
    "ave_pose_Tz",
    "med_mouth"]

x_variable_feature_all_colums = [
    # # gaze_x
    # "ave_gaze_x",
    # "std_gaze_x",
    # "max_gaze_x",
    # "min_gaze_x",
    # "med_gaze_x",
    # "skew_gaze_x",
    # "kurt_gaze_x",
    # # gaze_y
    # "ave_gaze_y",
    # "std_gaze_y",
    # "max_gaze_y",
    # "min_gaze_y",
    # "med_gaze_y",
    # "skew_gaze_y",
    # "kurt_gaze_y",
    # # gaze_hypo
    # "ave_gaze_hypo",
    # "std_gaze_hypo",
    # "max_gaze_hypo",
    # "min_gaze_hypo",
    # "med_gaze_hypo",
    # "skew_gaze_hypo",
    # "kurt_gaze_hypo",
    # # poze_Tx
    # "ave_pose_Tx",
    # "std_pose_Tx",
    # "max_pose_Tx",
    # "min_pose_Tx",
    # "med_pose_Tx",
    # "skew_pose_Tx",
    # "kurt_pose_Tx",
    # # poze_Ty
    # "ave_pose_Ty",
    # "std_pose_Ty",
    # "max_pose_Ty",
    # "min_pose_Ty",
    # "med_pose_Ty",
    # "skew_pose_Ty",
    # "kurt_pose_Ty",
    # # poze_Tz
    # "ave_pose_Tz",
    # "std_pose_Tz",
    # "max_pose_Tz",
    # "min_pose_Tz",
    # "med_pose_Tz",
    # "skew_pose_Tz",
    # "kurt_pose_Tz",
    # # poze_Rx
    # "ave_pose_Rx",
    # "std_pose_Rx",
    # "max_pose_Rx",
    # "min_pose_Rx",
    # "med_pose_Rx",
    # "skew_pose_Rx",
    # "kurt_pose_Rx",
    # # poze_Ry
    # "ave_pose_Ry",
    # "std_pose_Ry",
    # "max_pose_Ry",
    # "min_pose_Ry",
    # "med_pose_Ry",
    # "skew_pose_Ry",
    # "kurt_pose_Ry",
    # # poze_Rz
    # "ave_pose_Rz",
    # "std_pose_Rz",
    # "max_pose_Rz",
    # "min_pose_Rz",
    # "med_pose_Rz",
    # "skew_pose_Rz",
    # "kurt_pose_Rz",
    # # mouth
    # "ave_mouth",
    # "std_mouth",
    # "max_mouth",
    # "min_mouth",
    # "med_mouth",
    # "skew_mouth",
    # "kurt_mouth",
    # AU
    # "AU01_r_std", "AU01_r_max", "AU01_r_min", "AU01_r_median",  "AU01_r_p25",  "AU01_r_p75",
    # "AU02_r_std", "AU02_r_max", "AU02_r_min", "AU02_r_median",  "AU02_r_p25",  "AU02_r_p75",
    # "AU04_r_std", "AU04_r_max", "AU04_r_min", "AU04_r_median",  "AU04_r_p25",  "AU04_r_p75",
    # "AU05_r_std", "AU05_r_max", "AU05_r_min", "AU05_r_median",  "AU05_r_p25",  "AU05_r_p75",
    # "AU06_r_std", "AU06_r_max", "AU06_r_min", "AU06_r_median",  "AU06_r_p25",  "AU06_r_p75",
    # "AU07_r_std", "AU07_r_max", "AU07_r_min", "AU07_r_median",  "AU07_r_p25",  "AU07_r_p75",
    # "AU09_r_std", "AU09_r_max", "AU09_r_min", "AU09_r_median",  "AU09_r_p25",  "AU09_r_p75",
    # "AU10_r_std", "AU10_r_max", "AU10_r_min", "AU10_r_median",  "AU10_r_p25",  "AU10_r_p75",
    # "AU12_r_std", "AU12_r_max", "AU12_r_min", "AU12_r_median",  "AU12_r_p25",  "AU12_r_p75",
    # "AU14_r_std", "AU14_r_max", "AU14_r_min", "AU14_r_median",  "AU14_r_p25",  "AU14_r_p75",
    # "AU15_r_std", "AU15_r_max", "AU15_r_min", "AU15_r_median",  "AU15_r_p25",  "AU15_r_p75",
    # "AU17_r_std", "AU17_r_max", "AU17_r_min", "AU17_r_median",  "AU17_r_p25",  "AU17_r_p75",
    # "AU20_r_std", "AU20_r_max", "AU20_r_min", "AU20_r_median",  "AU20_r_p25",  "AU20_r_p75",
    # "AU23_r_std", "AU23_r_max", "AU23_r_min", "AU23_r_median",  "AU23_r_p25",  "AU23_r_p75",
    # "AU25_r_std", "AU25_r_max", "AU25_r_min", "AU25_r_median",  "AU25_r_p25",  "AU25_r_p75",
    # "AU26_r_std", "AU26_r_max", "AU26_r_min", "AU26_r_median",  "AU26_r_p25",  "AU26_r_p75",
    # "AU45_r_std", "AU45_r_max", "AU45_r_min", "AU45_r_median",  "AU45_r_p25",  "AU45_r_p75",
    # p25-75なし
    "AU01_r_std", "AU01_r_max", "AU01_r_min",
    "AU02_r_std", "AU02_r_max", "AU02_r_min",
    "AU04_r_std", "AU04_r_max", "AU04_r_min",
    "AU05_r_std", "AU05_r_max", "AU05_r_min",
    "AU06_r_std", "AU06_r_max", "AU06_r_min",
    "AU07_r_std", "AU07_r_max", "AU07_r_min",
    "AU09_r_std", "AU09_r_max", "AU09_r_min",
    "AU10_r_std", "AU10_r_max", "AU10_r_min",
    "AU12_r_std", "AU12_r_max", "AU12_r_min",
    "AU14_r_std", "AU14_r_max", "AU14_r_min",
    "AU15_r_std", "AU15_r_max", "AU15_r_min",
    "AU17_r_std", "AU17_r_max", "AU17_r_min",
    "AU20_r_std", "AU20_r_max", "AU20_r_min",
    "AU23_r_std", "AU23_r_max", "AU23_r_min",
    "AU25_r_std", "AU25_r_max", "AU25_r_min",
    "AU26_r_std", "AU26_r_max", "AU26_r_min",
    "AU45_r_std", "AU45_r_max", "AU45_r_min",
    # other_speak_status
    "isSpeak_other",


    # "isSpeak_other1",
    # "nonSpeak_other1", "startSpeak_other1", "speaking_other1", "endSpeak_other1",
    # "isSpeak_other2",
    # "nonSpeak_other2", "startSpeak_other2", "speaking_other2", "endSpeak_other2"
]


# 使わない
# --------------------------------------------------------------------

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
    "y", "y_pre_label", "y_pre_label_0.5s",  "y_pre_label_2s", "y_pre_label_3s", "y_pre_label_5s",
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
    # isSpeak
    "isSpeak_other",

    # "isSpeak_other1", "isSpeak_other2",
    # # speak status
    # "nonSpeak_other1", "nonSpeak_other2",
    # "startSpeak_other1", "startSpeak_other2",
    # "speaking_other1", "speaking_other2",
    # "endSpeak_other1", "endSpeak_other2"
]

# index変換用 カラム
feature_reindex_colums = [
    "y",
    "y_pre_label", "y_pre_label_0.5s",  "y_pre_label_2s", "y_pre_label_3s", "y_pre_label_5s",
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
    # isSpeak & speak status
    "isSpeak_other",
    # "isSpeak_other1",
    # "nonSpeak_other1", "startSpeak_other1", "speaking_other1", "endSpeak_other1",
    # "isSpeak_other2",
    # "nonSpeak_other2", "startSpeak_other2", "speaking_other2", "endSpeak_other2"
]


# モデル構築用 カラム
# 目的変数、説明変数の分離
x_variable_feature_colums = [
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

# action unit用

# load colums
columns_loading_AU = [
    " timestamp", " AU01_r", " AU02_r", " AU04_r", " AU05_r",
    " AU06_r", " AU07_r", " AU09_r", " AU10_r", " AU12_r", " AU14_r",
    " AU15_r", " AU17_r", " AU20_r", " AU23_r", " AU25_r", " AU26_r", " AU45_r"
]

columns_setting_header_AU = [
    " AU01_r", " AU02_r", " AU04_r", " AU05_r", " AU06_r",
    " AU07_r", " AU09_r", " AU10_r", " AU12_r", " AU14_r", " AU15_r",
    " AU17_r", " AU20_r", " AU23_r", " AU25_r", " AU26_r", " AU45_r"
]

columns_setting_pre_feature_header_AU = [
    " AU01_r", " AU02_r", " AU04_r", " AU05_r", " AU06_r",
    " AU07_r", " AU09_r", " AU10_r", " AU12_r", " AU14_r", " AU15_r",
    " AU17_r", " AU20_r", " AU23_r", " AU25_r", " AU26_r", " AU45_r", "y", "y_pre_label"
]

# カラム（出力）
feature_colums_rolling_AU = [
    "y", "y_pre_label",
    # std
    "AU01_r_std",
    "AU02_r_std",
    "AU04_r_std",
    "AU05_r_std",
    "AU06_r_std",
    "AU07_r_std",
    "AU09_r_std",
    "AU10_r_std",
    "AU12_r_std",
    "AU14_r_std",
    "AU15_r_std",
    "AU17_r_std",
    "AU20_r_std",
    "AU23_r_std",
    "AU25_r_std",
    "AU26_r_std",
    "AU45_r_std",
    # max
    "AU01_r_max",
    "AU02_r_max",
    "AU04_r_max",
    "AU05_r_max",
    "AU06_r_max",
    "AU07_r_max",
    "AU09_r_max",
    "AU10_r_max",
    "AU12_r_max",
    "AU14_r_max",
    "AU15_r_max",
    "AU17_r_max",
    "AU20_r_max",
    "AU23_r_max",
    "AU25_r_max",
    "AU26_r_max",
    "AU45_r_max",
    # min
    "AU01_r_min",
    "AU02_r_min",
    "AU04_r_min",
    "AU05_r_min",
    "AU06_r_min",
    "AU07_r_min",
    "AU09_r_min",
    "AU10_r_min",
    "AU12_r_min",
    "AU14_r_min",
    "AU15_r_min",
    "AU17_r_min",
    "AU20_r_min",
    "AU23_r_min",
    "AU25_r_min",
    "AU26_r_min",
    "AU45_r_min",
    # median
    "AU01_r_median",
    "AU02_r_median",
    "AU04_r_median",
    "AU05_r_median",
    "AU06_r_median",
    "AU07_r_median",
    "AU09_r_median",
    "AU10_r_median",
    "AU12_r_median",
    "AU14_r_median",
    "AU15_r_median",
    "AU17_r_median",
    "AU20_r_median",
    "AU23_r_median",
    "AU25_r_median",
    "AU26_r_median",
    "AU45_r_median",
    # p25
    "AU01_r_p25",
    "AU02_r_p25",
    "AU04_r_p25",
    "AU05_r_p25",
    "AU06_r_p25",
    "AU07_r_p25",
    "AU09_r_p25",
    "AU10_r_p25",
    "AU12_r_p25",
    "AU14_r_p25",
    "AU15_r_p25",
    "AU17_r_p25",
    "AU20_r_p25",
    "AU23_r_p25",
    "AU25_r_p25",
    "AU26_r_p25",
    "AU45_r_p25",
    # p75
    "AU01_r_p75",
    "AU02_r_p75",
    "AU04_r_p75",
    "AU05_r_p75",
    "AU06_r_p75",
    "AU07_r_p75",
    "AU09_r_p75",
    "AU10_r_p75",
    "AU12_r_p75",
    "AU14_r_p75",
    "AU15_r_p75",
    "AU17_r_p75",
    "AU20_r_p75",
    "AU23_r_p75",
    "AU25_r_p75",
    "AU26_r_p75",
    "AU45_r_p75",
]

# カラム（入れ替え）
feature_colums_reindex_AU = [
    "y_pre_label", "y",
    "AU01_r_std", "AU01_r_max", "AU01_r_min", "AU01_r_median",  "AU01_r_p25",  "AU01_r_p75",
    "AU02_r_std", "AU02_r_max", "AU02_r_min", "AU02_r_median",  "AU02_r_p25",  "AU02_r_p75",
    "AU04_r_std", "AU04_r_max", "AU04_r_min", "AU04_r_median",  "AU04_r_p25",  "AU04_r_p75",
    "AU05_r_std", "AU05_r_max", "AU05_r_min", "AU05_r_median",  "AU05_r_p25",  "AU05_r_p75",
    "AU06_r_std", "AU06_r_max", "AU06_r_min", "AU06_r_median",  "AU06_r_p25",  "AU06_r_p75",
    "AU07_r_std", "AU07_r_max", "AU07_r_min", "AU07_r_median",  "AU07_r_p25",  "AU07_r_p75",
    "AU09_r_std", "AU09_r_max", "AU09_r_min", "AU09_r_median",  "AU09_r_p25",  "AU09_r_p75",
    "AU10_r_std", "AU10_r_max", "AU10_r_min", "AU10_r_median",  "AU10_r_p25",  "AU10_r_p75",
    "AU12_r_std", "AU12_r_max", "AU12_r_min", "AU12_r_median",  "AU12_r_p25",  "AU12_r_p75",
    "AU14_r_std", "AU14_r_max", "AU14_r_min", "AU14_r_median",  "AU14_r_p25",  "AU14_r_p75",
    "AU15_r_std", "AU15_r_max", "AU15_r_min", "AU15_r_median",  "AU15_r_p25",  "AU15_r_p75",
    "AU17_r_std", "AU17_r_max", "AU17_r_min", "AU17_r_median",  "AU17_r_p25",  "AU17_r_p75",
    "AU20_r_std", "AU20_r_max", "AU20_r_min", "AU20_r_median",  "AU20_r_p25",  "AU20_r_p75",
    "AU23_r_std", "AU23_r_max", "AU23_r_min", "AU23_r_median",  "AU23_r_p25",  "AU23_r_p75",
    "AU25_r_std", "AU25_r_max", "AU25_r_min", "AU25_r_median",  "AU25_r_p25",  "AU25_r_p75",
    "AU26_r_std", "AU26_r_max", "AU26_r_min", "AU26_r_median",  "AU26_r_p25",  "AU26_r_p75",
    "AU45_r_std", "AU45_r_max", "AU45_r_min", "AU45_r_median",  "AU45_r_p25",  "AU45_r_p75",
]

# カラム（入れ替え）
x_variable_feature_colums_AU = [
    "y",
    "AU01_r_std", "AU01_r_max", "AU01_r_min",
    "AU02_r_std", "AU02_r_max", "AU02_r_min",
    "AU04_r_std", "AU04_r_max", "AU04_r_min",
    "AU05_r_std", "AU05_r_max", "AU05_r_min",
    "AU06_r_std", "AU06_r_max", "AU06_r_min",
    "AU07_r_std", "AU07_r_max", "AU07_r_min",
    "AU09_r_std", "AU09_r_max", "AU09_r_min",
    "AU10_r_std", "AU10_r_max", "AU10_r_min",
    "AU12_r_std", "AU12_r_max", "AU12_r_min",
    "AU14_r_std", "AU14_r_max", "AU14_r_min",
    "AU15_r_std", "AU15_r_max", "AU15_r_min",
    "AU17_r_std", "AU17_r_max", "AU17_r_min",
    "AU20_r_std", "AU20_r_max", "AU20_r_min",
    "AU23_r_std", "AU23_r_max", "AU23_r_min",
    "AU25_r_std", "AU25_r_max", "AU25_r_min",
    "AU26_r_std", "AU26_r_max", "AU26_r_min",
    "AU45_r_std", "AU45_r_max", "AU45_r_min"
]

column_au_p25_75 = [
    # median
    "AU01_r_median",
    "AU02_r_median",
    "AU04_r_median",
    "AU05_r_median",
    "AU06_r_median",
    "AU07_r_median",
    "AU09_r_median",
    "AU10_r_median",
    "AU12_r_median",
    "AU14_r_median",
    "AU15_r_median",
    "AU17_r_median",
    "AU20_r_median",
    "AU23_r_median",
    "AU25_r_median",
    "AU26_r_median",
    "AU45_r_median",
    # p25
    "AU01_r_p25",
    "AU02_r_p25",
    "AU04_r_p25",
    "AU05_r_p25",
    "AU06_r_p25",
    "AU07_r_p25",
    "AU09_r_p25",
    "AU10_r_p25",
    "AU12_r_p25",
    "AU14_r_p25",
    "AU15_r_p25",
    "AU17_r_p25",
    "AU20_r_p25",
    "AU23_r_p25",
    "AU25_r_p25",
    "AU26_r_p25",
    "AU45_r_p25",
    # p75
    "AU01_r_p75",
    "AU02_r_p75",
    "AU04_r_p75",
    "AU05_r_p75",
    "AU06_r_p75",
    "AU07_r_p75",
    "AU09_r_p75",
    "AU10_r_p75",
    "AU12_r_p75",
    "AU14_r_p75",
    "AU15_r_p75",
    "AU17_r_p75",
    "AU20_r_p75",
    "AU23_r_p75",
    "AU25_r_p75",
    "AU26_r_p75",
    "AU45_r_p75"
]
