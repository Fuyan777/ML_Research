from pprint import pprint
from numpy.core.defchararray import array
import pandas as pd
import matplotlib.pyplot as plt
import file_path
import warnings
import common
import numpy as np

# SettingWithCopyWarningの非表示
warnings.simplefilter("ignore")

#
# パラメータ変更
#

# 被験者の種類
user = "b"

# ウィンドウサイズの設定w（0.033 : 0.5秒先=15, 1秒=30, 2秒=60, 3秒=90, 5秒=150 ）

# ウィンドウサイズの設定w（0.083 : 0.5秒先=6, 1秒=12, 2秒=24, 3秒=36, 5秒=60 ）
window_size = 150

# 予測フレームシフトの設定s（ 0.5秒先=6, 1秒=12, 2秒=24, 3秒=36, 5秒=60 ）
pre_speak_time = 15

# 予測時間の設定
speak = "5w1s"

# サンプル数を揃える
speak_data_count = 500

# 顔特徴csvのpath設定
face_data_path = "b-20210128"

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
    extraction_feature_value_v2()


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
    print(df_feature.index)
    print(df_feature[df_feature["y"] == 0].index)
    print(df_feature[df_feature["y"] == 1].index)  # ここの行数に合わせる必要あり

    # csvに書き込み
    df_feature.to_csv(
        file_path.face_feature_csv + "/%s-feature/pre-feat_val_%s.csv" % (user, speak),
        mode="w",  # 上書き
        # header=False,
        index=False,
    )


#
# 特徴量の抽出ver2.0（先にウィンドウ処理）
#


def extraction_feature_value_v2():
    df_face = pd.read_csv(
        file_path.face_feature_csv + "/%s-feature/pre-feat_val_%s.csv" % (user, speak),
    )

    spk_data = df_window_v2(window_size, df_face)
    print(spk_data)

    # 特徴量をcsvに書き込み

    # spk_data.to_csv(
    #     file_path.face_feature_csv + "/%s-feature/feature_value/test.csv" % (user),
    #     mode="w",  # 上書き
    #     index=False,
    # )

    spk_data.to_csv(
        file_path.face_feature_csv
        + "/%s-feature/feature_value/feat_val_%s.csv" % (user, speak),
        mode="w",  # 上書き
        index=False,
    )


#
# ウィンドウ処理ver2.0
#

feature_list = [
    " gaze_angle_x",
    " gaze_angle_y",
    " pose_Tx",
    " pose_Ty",
    " pose_Tz",
    " pose_Rx",
    " pose_Ry",
    " pose_Rz",
    "mouth",
]


def df_window_v2(window_size, df_feature):
    print("=========ウィンドウ処理開始==============")
    # ウインドウ処理
    df_y = (
        df_feature["y"]
        .shift(shift_size)
        .rolling(window_size, min_periods=1)
        .apply(judge_y)
    )

    df_y_pre = (
        df_feature["y_pre_label"]
        .shift(shift_size)
        .rolling(window_size, min_periods=1)
        .apply(judge_y_pre)
    )

    print("=========ウィンドウ処理（mean）==============")
    df_ave = round(
        df_feature[feature_list]
        .shift(shift_size)
        .rolling(window_size, min_periods=1)
        .apply(judge_speak_mean),
        3,
    )
    print("=========ウィンドウ処理（std）==============")
    df_std = round(
        df_feature[feature_list]
        .shift(shift_size)
        .rolling(window_size, min_periods=1)
        .apply(judge_speak_std),
        3,
    )
    print("=========ウィンドウ処理（max）==============")
    df_max = round(
        df_feature[feature_list]
        .shift(shift_size)
        .rolling(window_size, min_periods=1)
        .apply(judge_speak_max),
        3,
    )
    print("=========ウィンドウ処理（min）==============")
    df_min = round(
        df_feature[feature_list]
        .shift(shift_size)
        .rolling(window_size, min_periods=1)
        .apply(judge_speak_min),
        3,
    )
    print("=========ウィンドウ処理（med）==============")
    df_med = round(
        df_feature[feature_list]
        .shift(shift_size)
        .rolling(window_size, min_periods=1)
        .apply(judge_speak_median),
        3,
    )
    print("=========ウィンドウ処理（skew）==============")
    df_skew = round(
        df_feature[feature_list]
        .shift(shift_size)
        .rolling(window_size, min_periods=1)
        .apply(judge_speak_skew),
        3,
    )
    print("=========ウィンドウ処理（kurt）==============")
    df_kurt = round(
        df_feature[feature_list]
        .shift(shift_size)
        .rolling(window_size, min_periods=1)
        .apply(judge_speak_kurt),
        3,
    )
    print("=========ウィンドウ処理終了==============")

    # dfの結合
    tmp_all_feature = pd.concat(
        [
            df_y,
            df_y_pre,
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
    print("----------ウィンドウ処理後---------")
    print(df_all_feature)

    # カラムソート
    df_all_feature_sorted = df_all_feature.reindex(
        columns=common.feature_colums_reindex
    )

    return df_all_feature_sorted


def judge_speak_mean(array_value):
    return array_value.mean()


def judge_speak_std(array_value):
    return array_value.std()


def judge_speak_min(array_value):
    return array_value.min()


def judge_speak_max(array_value):
    return array_value.max()


def judge_speak_median(array_value):
    return array_value.median()


def judge_speak_skew(array_value):
    return array_value.skew()


def judge_speak_kurt(array_value):
    return array_value.kurt()


def judge_y(array_value):
    if any(array_value.isin([np.nan])):
        return np.nan
    elif any(array_value.isin([0.0])):
        return np.nan
    else:  # 発話区間も含む場合、else部分は消す
        return array_value.iloc[-1]  # 配列の最後の要素を返す


def judge_y_pre(array_value):
    return array_value.iloc[-1]


#
# 特徴量の抽出ver1.0（先に発話区間の除去）
#


def extraction_feature_value():
    df_face = pd.read_csv(
        file_path.face_feature_csv + "/%s-feature/pre-feat_val_%s.csv" % (user, speak),
    )

    # 非発話区間のデータのみ抽出
    df_focus_y = df_face[(df_face["y"] == 1)]

    # 発話時の特徴量
    df_focus_y_spk = df_focus_y[(df_focus_y["y_pre_label"] == 0)]
    df_focus_y_spk_dropped = df_focus_y_spk.drop(["y", "y_pre_label"], axis=1)

    # 非発話時の特徴量
    df_focus_y_non = df_focus_y[(df_focus_y["y_pre_label"] == 1)]
    df_focus_y_non_dropped = df_focus_y_non.drop(["y", "y_pre_label"], axis=1)

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
