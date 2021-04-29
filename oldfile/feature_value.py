import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib
import pandas as pd
import csv
import warnings
from pprint import pprint
import file_path
import Common

# SettingWithCopyWarningの非表示
warnings.simplefilter("ignore")

# グラフfont
mpl.rcParams["font.family"] = "Times New Roman"

# 口の開き具合算出


def calc_mouth(mouth_top, mouth_bottom):
    mouth_open = mouth_bottom - mouth_top
    return mouth_open


# 各顔特徴データのグラフ作成


def showGragh(df_plot, cnt, face):
    plt.figure()
    df_plot.plot()
    plt.ylabel("出力値", fontname="AppleGothic")
    plt.xlabel("秒[sec]", fontname="AppleGothic")
    plt.ylim([0, 30])
    # plt.show()
    plt.savefig(
        "/Users/fuyan/Documents/siraisi_lab/B4/40_program/graph/%s/img%d.png"
        % (face, cnt)
    )
    plt.close("all")


# パス設定
date = "20201015"
user = "c"
# user = 'c'
# speak = 'keep'
# speak = 'change'
speak = "non"

# 特徴量の抽出
# turn-keepかtakingでpathを変える必要あり
time_speak_path = file_path.csv_path + "/speak-20201015/face_%s_speak_%s.csv" % (
    user,
    speak,
)

# OpenFaceによる顔特徴データ
# face_feature_path = "/Users/fuyan/research/OpenFace/processed/%s-%s.csv" % (user, date)
face_feature_path = "/Users/fuyan/Documents/c-20201015.csv"
start_time = []
end_time = []

# 発話・非発話区間・発話タイミングのデータ抽出処理（秒単位）
df_sp = pd.read_csv(time_speak_path)
start_time = df_sp["start"]
end_time = df_sp["end"]
interval_time = df_sp["end"] - df_sp["start"]  # 発話間隔

# print('{}\n{}\n{}\n'.format(start_time, end_time, interval_time))

# 顔特徴データから特徴量の抽出処理
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
        " y_62",
        " y_66",
    ],
)
cnt = 0

# 顔特徴のデータ格納
for speak_index in df_sp.index:
    timestamp_d = df_face[
        (df_face[" timestamp"] > start_time[speak_index])
        & (df_face[" timestamp"] < end_time[speak_index])
    ]
    timestamp_d["mouth"] = timestamp_d[" y_66"] - timestamp_d[" y_62"]  # 口の開き具合

    feature_value.append(round(timestamp_d.mean(), 3))  # 平均
    feature_value.append(round(timestamp_d.std(), 3))  # 標準偏差
    feature_value.append(round(timestamp_d.min(), 3))  # 最小値
    feature_value.append(round(timestamp_d.max(), 3))  # 最大値
    feature_value.append(round(timestamp_d.median(), 3))  # 中央値

# 使用しない列の削除
df = pd.DataFrame(feature_value)
df_drop = df.drop([" timestamp", " y_62", " y_66"], axis=1)
feature_value_t = df_drop.T.values.tolist()

# 非発話区間ごとの特徴量を格納
feature_list = []
feature_array = []
feature_count = 0

for i in range(len(feature_value_t)):
    feature_count = 0

    for j in range(len(feature_value_t[i])):
        if i == 0:
            feature_array.append(feature_value_t[i][j])
            if (j + 1) % 5 == 0:
                feature_list.append(feature_array)
                feature_array = []
        else:
            feature_list[feature_count].append(feature_value_t[i][j])
            if (j + 1) % 5 == 0:
                feature_count += 1

feature_value_list = []
for i in range(len(feature_list)):
    feature_value_list.append(feature_list[i])

with open(
    file_path.csv_path + "/%s-output_feature_value_%s.csv" % (user, speak),
    "w",
) as f:
    writer = csv.writer(f)
    writer.writerow(Common.feature_colums)
    writer.writerows(feature_value_list)

# 特徴量csvの読み取り
feature_csv = pd.read_csv(
    file_path.csv_path + "/%s-output_feature_value_%s.csv" % (user, speak)
)
print(feature_csv)
