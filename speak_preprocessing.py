from pprint import pprint
import pandas as pd
import matplotlib.pyplot as plt
import file_path
import warnings
import common

# SettingWithCopyWarningの非表示
warnings.simplefilter("ignore")

#
# パラメータ変更
#

# 被験者の種類
user = "a"

# ウィンドウサイズの設定w（0.033 : 0.5秒先=15, 1秒=30, 2秒=60, 3秒=90, 5秒=150 ）

# ウィンドウサイズの設定w（0.083 : 0.5秒先=6, 1秒=12, 2秒=24, 3秒=36, 5秒=60 ）
window_size = 4

# 予測フレームシフトの設定s（ 0.5秒先=6, 1秒=12, 2秒=24, 3秒=36, 5秒=60 ）
pre_speak_time = 6

# 予測時間の設定
speak = "0.1w_0.5s"

# サンプル数を揃える
speak_data_count = 1000

# 顔特徴csvのpath設定
face_data_path = "a-20210128"

# overlapの計算
shift_size = (window_size // 2) - 1

columns = [
    " gaze_angle_x",
    " gaze_angle_y",
    " pose_Tx",
    " pose_Ty",
    " pose_Tz",
    " pose_Rx",
    " pose_Ry",
    " pose_Rz",
    "mouth",
    "y",
]


def main():
    start_speak = []
    end_speak = []
    speak_label = []  # 発話：x、非発話：s

    speak_data = extraction_speak_data()

    for speak in speak_data:
        start_speak.append(float(speak[0]))
        end_speak.append(float(speak[1]))
        speak_label.append(speak[2])

    # 会話データ作成
    label_face(speak_label, start_speak, end_speak)
    # 特徴量の抽出
    extraction_feature_value()


#
# 発話・非発話の時間を抽出
#


def extraction_speak_data():
    f = open("elan_output_txt/%s.txt" % (face_data_path), "r", encoding="UTF-8")
    tmp_data = []
    datalines = f.readlines()

    for data in datalines:
        tmp_data.append(data.split())

    f.close()
    return tmp_data


#
# 顔特徴データのラベリング
#


def label_face(label, start_time, end_time):
    face_feature = pd.read_csv(file_path.face_feature_path + face_data_path + ".csv")

    # 誤認識は全て削除
    face_feature_dropped = face_feature[face_feature[" success"] == 1]
    df_face = pd.DataFrame(
        face_feature_dropped,
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

    df_header = pd.DataFrame(columns=columns)

    for index in range(len(label)):
        # 各非発話区間ごとの顔特徴データ
        ts_df = df_face[
            (df_face[" timestamp"] >= start_time[index])
            & (df_face[" timestamp"] <= end_time[index])
        ]

        # 口の開き具合
        ts_df["mouth"] = ts_df[" y_66"].copy() - ts_df[" y_62"].copy()
        ts_df["y"] = 0 if label[index] == "x" else 1
        ts_df_feature = ts_df.drop([" timestamp", " y_62", " y_66"], axis=1)

        # 少数第三まで
        df_feature = round(ts_df_feature, 3)

        # 縦に結合
        df_header = pd.concat([df_header, df_feature])

    # 数秒先をラベリング
    df_header["y_pre_label"] = df_header["y"].shift(-pre_speak_time)
    df_feature = df_header.dropna()

    # csvに書き込み
    df_feature.to_csv(
        file_path.face_feature_csv + "/%s-feature/pre-feat_val_%s.csv" % (user, speak),
        mode="w",  # 上書き
        # header=False,
        index=False,
    )


#
# 特徴量の抽出
#


def extraction_feature_value():
    df_face = pd.read_csv(
        file_path.face_feature_csv + "/%s-feature/pre-feat_val_%s.csv" % (user, speak),
    )

    df_focus_y = df_face[(df_face["y"] == 1)]

    # 発話時の特徴量
    df_focus_y_spk = df_focus_y[(df_focus_y["y_pre_label"] == 0)]
    df_focus_y_spk_dropped = df_focus_y_spk.drop(["y", "y_pre_label"], axis=1)

    # 非発話時の特徴量
    df_focus_y_non = df_focus_y[(df_focus_y["y_pre_label"] == 1)]
    df_focus_y_non_dropped = df_focus_y_non.drop(["y", "y_pre_label"], axis=1)

    # 　予備動作区間の切り出し（グラフ化）
    # generate_timedata_graph(df_focus_y_spk_dropped, shift_size, window_size, 0)
    # generate_timedata_graph(df_focus_y_non_dropped, shift_size, window_size, 1)

    # ウィンドウ処理
    spk_data = df_window(window_size, df_focus_y_spk_dropped)
    non_data = df_window(window_size, df_focus_y_non_dropped)
    print(len(spk_data.index))
    print(len(non_data.index))

    spk_data["y"] = 0
    non_data["y"] = 1
    spk_data_delete = spk_data.head(speak_data_count)
    non_data_delete = non_data.head(speak_data_count)

    # 両ラベル結合
    tmp_all_feature = pd.concat([spk_data_delete, non_data_delete])

    print(tmp_all_feature)

    # 特徴量をcsvに書き込み
    tmp_all_feature.to_csv(
        file_path.face_feature_csv
        + "/%s-feature/feature_value/feat_val_%s.csv" % (user, speak),
        mode="w",  # 上書き
        index=False,
    )


#
# ウィンドウ処理
#


def df_window(window_size, df_feature):
    # ウインドウ処理
    df_ave = round(
        df_feature.shift(shift_size).rolling(window_size, min_periods=1).mean(), 3
    )
    df_std = round(
        df_feature.shift(shift_size).rolling(window_size, min_periods=1).std(), 3
    )
    df_max = round(
        df_feature.shift(shift_size).rolling(window_size, min_periods=1).max(), 3
    )
    df_min = round(
        df_feature.shift(shift_size).rolling(window_size, min_periods=1).min(), 3
    )
    df_med = round(
        df_feature.shift(shift_size).rolling(window_size, min_periods=1).median(), 3
    )
    df_skew = round(
        df_feature.shift(shift_size).rolling(window_size, min_periods=1).skew(), 3
    )
    df_kurt = round(
        df_feature.shift(shift_size).rolling(window_size, min_periods=1).kurt(), 3
    )

    # overlapなし
    # df_ave = round(df_feature.rolling(window_size, min_periods=1).mean(), 3)
    # df_std = round(df_feature.rolling(window_size, min_periods=1).std(), 3)
    # df_max = round(df_feature.rolling(window_size, min_periods=1).max(), 3)
    # df_min = round(df_feature.rolling(window_size, min_periods=1).min(), 3)
    # df_med = round(df_feature.rolling(window_size, min_periods=1).median(), 3)
    # df_skew = round(df_feature.rolling(window_size, min_periods=1).skew(), 3)
    # df_kurt = round(df_feature.rolling(window_size, min_periods=1).kurt(), 3)

    # dfの結合
    tmp_all_feature = pd.concat(
        [
            df_ave,
            df_std,
            df_max,
            df_min,
            df_med,
            df_skew,
            df_kurt,
        ],
        axis=1,
    )
    df_all_feature_drop = tmp_all_feature.dropna()
    df_all_feature = df_all_feature_drop.set_axis(common.feature_rolling_colums, axis=1)

    # カラムソート
    df_all_feature_sorted = df_all_feature.reindex(
        columns=common.feature_colums_reindex
    )

    return df_all_feature_sorted


#
# 時系列データの切り出しグラフの生成
#


def generate_timedata_graph(df_feature, shift_size, window_size, flag):
    # スライスの位置設定
    start = 0
    end = window_size - 1

    for index in range(len(df_feature.index)):
        if index == 10:
            break

        plt.figure()

        # ラベル設定
        plt.ylabel("出力値", fontname="AppleGothic")
        plt.xlabel("frame", fontname="AppleGothic")

        df_feature_slice = df_feature[start:end]
        df_select_face = df_feature_slice["mouth"]

        # 出力値の範囲
        # plt.ylim([df_select_face.min() - 1, df_select_face.max() + 1])
        plt.ylim([0, 30])

        # 繰り上げ処理（overlap: 0%）
        start += window_size
        end += window_size

        # 繰り上げ処理（overlap: 50%）
        # start += shift_size
        # end += shift_size

        df_select_face.plot()
        isSpk = "spk" if flag == 0 else "non"
        plt.savefig(
            file_path.path
            + "ml_graph/%s-graph/%s_timedata_w=5/timedata%d.png" % (user, isSpk, index)
        )

        plt.close("all")


if __name__ == "__main__":
    main()

    # print("********************************************")
    # print("********************************************")
    # print("********************************************")

    # print("---------start / end----------")
    # print(start)
    # print(end)
    # print("--------------------------------")

    # print("------df_feature_slice------")
    # print(df_feature_slice)
    # print("----------------------------")
