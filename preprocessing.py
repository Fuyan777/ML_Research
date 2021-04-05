import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib
import pandas as pd
import csv
import warnings
from pprint import pprint
import file_path
import common

# SettingWithCopyWarningの非表示
warnings.simplefilter("ignore")

# パス設定
date = "20210115"
user = "a"
speak = "keep-v2"
# speak = 'change'
# speak = "non"

# 特徴量の抽出
# turn-keepかtakingでpathを変える必要あり
time_speak_path = file_path.csv_path + "/speak-20210115/face_%s_speak_%s.csv" % (
    user,
    speak,
)

#
# 発話・非発話区間・発話タイミングのデータ抽出処理（秒単位）
#

df_sp = pd.read_csv(time_speak_path)
start_time = df_sp["start"]
end_time = df_sp["end"]

#
# OpenFaceによる顔特徴データのpath
#

# face_feature_path = "/Users/fuyan/research/OpenFace/processed/%s-%s.csv" % (user, date)
face_feature_path = "/Users/fuyan/Documents/a-20200115-2.csv"

#
# 顔特徴データから特徴量の抽出処理
#

feature_value = []
feature = []
face_feature = pd.read_csv(face_feature_path)
df_face = pd.DataFrame(
    face_feature,
    columns=[
        " timestamp",
        " gaze_angle_x",
        " gaze_angle_y",
        " pose_Tx",
        " pose_Ty",
        " pose_Tz",
        " pose_Rx",
        " pose_Ry",
        " pose_Rz",
        " y_62",
        " y_66",
    ],
)

windo_size = 4

# 非発話区間の顔特徴データを抽出
for speak_index in df_sp.index:
    # 各非発話区間ごとの顔特徴データ
    ts_df = df_face[
        (df_face[" timestamp"] > start_time[speak_index])
        & (df_face[" timestamp"] < end_time[speak_index])
    ]

    # 口の開き具合
    ts_df["mouth"] = ts_df[" y_66"].copy() - ts_df[" y_62"].copy()
    ts_df_feature = ts_df.drop([" timestamp", " y_62", " y_66"], axis=1)

    # ウィンドウ処理
    df_ave = round(ts_df_feature.rolling(windo_size, min_periods=1).mean(), 3)
    df_std = round(ts_df_feature.rolling(windo_size, min_periods=1).std(), 3)
    df_max = round(ts_df_feature.rolling(windo_size, min_periods=1).max(), 3)
    df_min = round(ts_df_feature.rolling(windo_size, min_periods=1).min(), 3)
    df_med = round(ts_df_feature.rolling(windo_size, min_periods=1).median(), 3)
    df_skew = round(ts_df_feature.rolling(windo_size, min_periods=1).skew(), 3)
    df_kurt = round(ts_df_feature.rolling(windo_size, min_periods=1).kurt(), 3)

    # 全ての特徴量の列を結合
    tmp_all_feature = pd.concat(
        [df_ave, df_std, df_max, df_min, df_med, df_skew, df_kurt],
        axis=1,
    )
    df_all_feature_drop = tmp_all_feature.dropna()
    df_all_feature = df_all_feature_drop.set_axis(common.feature_rolling_colums, axis=1)

    # カラムソート
    df_all_feature_sorted = df_all_feature.reindex(
        columns=common.feature_colums_reindex
    )

    # csvに書き込み
    df_all_feature_sorted.to_csv(
        file_path.csv_path + "/%s-output_feature_value_%s.csv" % (user, speak),
        mode="a",  # 上書き
        header=False,
    )

df_header = pd.DataFrame(columns=common.feature_colums_reindex)
df_header.to_csv(file_path.csv_path + "/header.csv")