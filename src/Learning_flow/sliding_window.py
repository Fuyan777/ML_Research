# learning module
from resources import resources

# external module
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

feature_list = [
    " gaze_angle_x",
    " gaze_angle_y",
    "gaze_angle_hypo",
    " pose_Tx",
    " pose_Ty",
    " pose_Tz",
    " pose_Rx",
    " pose_Ry",
    " pose_Rz",
    "mouth",
]

feature_list_user = ["isSpeak_other1", "isSpeak_other2"]

feature_list_AU = [
    " AU01_r",
    " AU02_r",
    " AU04_r",
    " AU05_r",
    " AU06_r",
    " AU07_r",
    " AU09_r",
    " AU10_r",
    " AU12_r",
    " AU14_r",
    " AU15_r",
    " AU17_r",
    " AU20_r",
    " AU23_r",
    " AU25_r",
    " AU26_r",
    " AU45_r",
]


class SlidingWindow():
    #
    # ウィンドウ処理
    #

    def run(
        self,
        window_size,
        df_feature_value,
    ):
        print(df_feature_value)
        # overlapの計算
        shift_size = (window_size // 2) - 1
        print("=========ウィンドウ処理（label）==============")
        # ウインドウ処理
        df_y = (
            df_feature_value["y"]
            .shift(shift_size)
            .rolling(window_size, min_periods=1)
            .apply(judge_y)
        )

        df_y_pre = (
            df_feature_value["y_pre_label"]
            .shift(shift_size)
            .rolling(window_size, min_periods=1)
            .apply(judge_y_pre)
        )
        df_y_pre_05s = (
            df_feature_value["y_pre_label_2s"]
            .shift(shift_size)
            .rolling(window_size, min_periods=1)
            .apply(judge_y_pre)
        )
        df_y_pre_2s = (
            df_feature_value["y_pre_label_2s"]
            .shift(shift_size)
            .rolling(window_size, min_periods=1)
            .apply(judge_y_pre)
        )
        df_y_pre_3s = (
            df_feature_value["y_pre_label_3s"]
            .shift(shift_size)
            .rolling(window_size, min_periods=1)
            .apply(judge_y_pre)
        )
        df_y_pre_5s = (
            df_feature_value["y_pre_label_5s"]
            .shift(shift_size)
            .rolling(window_size, min_periods=1)
            .apply(judge_y_pre)
        )

        print("=========ウィンドウ処理（mean）==============")
        df_ave = round(
            df_feature_value[feature_list]
            .shift(shift_size)
            .rolling(window_size, min_periods=1)
            .apply(judge_speak_mean),
            3,
        )
        print("=========ウィンドウ処理（std）==============")
        df_std = round(
            df_feature_value[feature_list]
            .shift(shift_size)
            .rolling(window_size, min_periods=1)
            .apply(judge_speak_std),
            3,
        )
        print("=========ウィンドウ処理（max）==============")
        df_max = round(
            df_feature_value[feature_list]
            .shift(shift_size)
            .rolling(window_size, min_periods=1)
            .apply(judge_speak_max),
            3,
        )
        print("=========ウィンドウ処理（min）==============")
        df_min = round(
            df_feature_value[feature_list]
            .shift(shift_size)
            .rolling(window_size, min_periods=1)
            .apply(judge_speak_min),
            3,
        )
        print("=========ウィンドウ処理（med）==============")
        df_med = round(
            df_feature_value[feature_list]
            .shift(shift_size)
            .rolling(window_size, min_periods=1)
            .apply(judge_speak_median),
            3,
        )
        print("=========ウィンドウ処理（skew）==============")
        df_skew = round(
            df_feature_value[feature_list]
            .shift(shift_size)
            .rolling(window_size, min_periods=1)
            .apply(judge_speak_skew),
            3,
        )
        print("=========ウィンドウ処理（kurt）==============")
        df_kurt = round(
            df_feature_value[feature_list]
            .shift(shift_size)
            .rolling(window_size, min_periods=1)
            .apply(judge_speak_kurt),
            3,
        )
        print("=========ウィンドウ処理（isSpeak_other）==============")
        # df_other_speak1 = (
        #     df_feature_value["isSpeak_other1"]
        #     .shift(shift_size)
        #     .rolling(window_size, min_periods=1)
        #     .apply(judge_is_speak_other)
        # )

        # df_other_speak2 = (
        #     df_feature_value["isSpeak_other2"]
        #     .shift(shift_size)
        #     .rolling(window_size, min_periods=1)
        #     .apply(judge_is_speak_other)
        # )

        df_other_speak = (
            df_feature_value["isSpeak_other"]
            .shift(shift_size)
            .rolling(window_size, min_periods=1)
            .apply(judge_is_speak_other)
        )

        df_duration_of_other_speak = (
            df_feature_value["duration_of_speak_other"]
            .shift(shift_size)
            .rolling(window_size, min_periods=1)
            .apply(judge_duration_of_other_speak)
        )

        df_duration_of_other_non_speak = (
            df_feature_value["duration_of_non_speak_other"]
            .shift(shift_size)
            .rolling(window_size, min_periods=1)
            .apply(judge_duration_of_other_speak)
        )
        # print("=========ウィンドウ処理（speak status）==============")
        # df_other_non_speak = (
        #     df_feature_value[feature_list_user]
        #     .shift(shift_size)
        #     .rolling(window_size, min_periods=1)
        #     .apply(judge_nonSpeak)
        # )
        # df_other_start_speak = (
        #     df_feature_value[feature_list_user]
        #     .shift(shift_size)
        #     .rolling(window_size, min_periods=1)
        #     .apply(judge_startSpeak)
        # )
        # df_other_speaking = (
        #     df_feature_value[feature_list_user]
        #     .shift(shift_size)
        #     .rolling(window_size, min_periods=1)
        #     .apply(judge_speaking)
        # )
        # df_other_end_speak = (
        #     df_feature_value[feature_list_user]
        #     .shift(shift_size)
        #     .rolling(window_size, min_periods=1)
        #     .apply(judge_endSpeak)
        # )
        print("=========後処理==============")
        # dfの結合
        tmp_all_feature = pd.concat(
            [
                df_y,
                df_y_pre,
                df_y_pre_05s,
                df_y_pre_2s,
                df_y_pre_3s,
                df_y_pre_5s,
                df_ave,
                df_std,
                df_max,
                df_min,
                df_med,
                df_skew,
                df_kurt,
                df_other_speak,
                df_duration_of_other_speak,
                df_duration_of_other_non_speak,
                # df_other_speak1,
                # df_other_speak2,
                # df_other_non_speak,
                # df_other_start_speak,
                # df_other_speaking,
                # df_other_end_speak
            ],
            axis=1,
        )

        # 本来のカラム名と値を合わせる
        df_all_feature = tmp_all_feature.set_axis(
            resources.feature_rolling_colums,
            axis=1
        )

        df_all_feature_drop = df_all_feature.dropna()

        # カラムのソート
        df_reindex = df_all_feature_drop.reindex(
            resources.feature_reindex_colums,
            axis=1
        )
        return df_reindex

    def run_AU(
        self,
        window_size,
        df_feature_value,
    ):
        # overlapの計算
        shift_size = (window_size // 2) - 1

        print(df_feature_value)
        # ウインドウ処理
        df_y = (
            df_feature_value["y"]
            .shift(shift_size)
            .rolling(window_size, min_periods=1)
            .apply(judge_y)
        )

        df_y_pre = (
            df_feature_value["y_pre_label"]
            .shift(shift_size)
            .rolling(window_size, min_periods=1)
            .apply(judge_y_pre)
        )
        print("=========ウィンドウ処理（std）==============")
        df_std = round(
            df_feature_value[feature_list_AU]
            .shift(shift_size)
            .rolling(window_size, min_periods=1)
            .apply(judge_speak_std),
            3,
        )
        print("=========ウィンドウ処理（max）==============")
        df_max = round(
            df_feature_value[feature_list_AU]
            .shift(shift_size)
            .rolling(window_size, min_periods=1)
            .apply(judge_speak_max),
            3,
        )
        print("=========ウィンドウ処理（min）==============")
        df_min = round(
            df_feature_value[feature_list_AU]
            .shift(shift_size)
            .rolling(window_size, min_periods=1)
            .apply(judge_speak_min),
            3,
        )
        print("=========ウィンドウ処理（median）==============")
        df_median = round(
            df_feature_value[feature_list_AU]
            .shift(shift_size)
            .rolling(window_size, min_periods=1)
            .apply(judge_speak_median),
            3,
        )
        print("=========ウィンドウ処理（p25）==============")
        df_p25 = round(
            df_feature_value[feature_list_AU]
            .shift(shift_size)
            .rolling(window_size, min_periods=1)
            .apply(judge_speak_p25),
            3
        )
        print("=========ウィンドウ処理（p75）==============")
        df_p75 = round(
            df_feature_value[feature_list_AU]
            .shift(shift_size)
            .rolling(window_size, min_periods=1)
            .apply(judge_speak_p75),
            3
        )

        # dfの結合
        tmp_all_feature = pd.concat(
            [
                df_y,
                df_y_pre,
                df_std,
                df_max,
                df_min,
                df_median,
                df_p25,
                df_p75
            ],
            axis=1,
        )

        print(tmp_all_feature)

        # 本来のカラム名と値を合わせる
        df_all_feature = tmp_all_feature.set_axis(
            resources.feature_colums_rolling_AU,
            axis=1
        )

        df_all_feature_drop = df_all_feature.dropna()

        # カラムのソート
        df_reindex = df_all_feature_drop.reindex(
            resources.feature_colums_reindex_AU,
            axis=1
        )
        return df_reindex

    # median, p25, p75の追加用
    def add_au_feature(self, window_size, df_feature_value):
        # overlapの計算
        shift_size = (window_size // 2) - 1

        print("=========ウィンドウ処理（median）==============")
        df_median = round(
            df_feature_value[feature_list_AU]
            .shift(shift_size)
            .rolling(window_size, min_periods=1)
            .apply(judge_speak_median),
            3,
        )
        print("=========ウィンドウ処理（p25）==============")
        df_p25 = round(
            df_feature_value[feature_list_AU]
            .shift(shift_size)
            .rolling(window_size, min_periods=1)
            .apply(judge_speak_p25),
            3
        )
        print("=========ウィンドウ処理（p75）==============")
        df_p75 = round(
            df_feature_value[feature_list_AU]
            .shift(shift_size)
            .rolling(window_size, min_periods=1)
            .apply(judge_speak_p75),
            3
        )

        # dfの結合
        tmp_all_feature = pd.concat(
            [
                df_median,
                df_p25,
                df_p75
            ],
            axis=1,
        )
        df_all_feature = tmp_all_feature.set_axis(
            resources.feature_colums_rolling_AU,
            axis=1
        )

        df_all_feature_drop = df_all_feature.dropna()

        return df_all_feature_drop


# sliding window method


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


def judge_speak_p25(array_value):  # 25パーセンタイル
    return array_value.quantile(0.25)


def judge_speak_p75(array_value):  # 75パーセンタイル
    return array_value.quantile(0.75)


# 発話データを除外する処理


# 発話ラベル
speak_value = 0.0
# 非発話ラベル
non_speak_value = 1.0

# 処理内容 : 配列内に発話が含めば、nanを返す


def judge_y(array_value):
    if any(array_value.isin([np.nan])):
        return np.nan
    elif any(array_value.isin([speak_value])):
        return np.nan
    else:  # 発話区間も含む場合、else部分は消す
        return array_value.iloc[-1]  # 配列の最後の要素を返す


def judge_y_pre(array_value):
    return array_value.iloc[-1]  # 配列の最後の要素を返す


# 発話の有無


def judge_is_speak_other(array_value):
    if any(array_value.isin([speak_value])):
        return speak_value
    else:
        return non_speak_value

# 発話継続時間
def judge_duration_of_other_speak(array_value):
    # 配列の最後の要素が0 (発話 or 非発話) の時は時間をリセット
    if array_value.iloc[-1] == 0:
        return 0
    else:
        return array_value.sum()

# 発話状態
# 非発話


def judge_nonSpeak(array_value):
    # ウィンドウ内に全て非発話ラベルであれば1、それ以外0
    if all(array_value.isin([non_speak_value])):
        return 1
    else:
        return 0


def judge_startSpeak(array_value):
    # ウィンドウ内の前半部分に1が含まれている
    window_length = round(len(array_value)/2)
    if any(array_value[:window_length].isin([non_speak_value])) and all(array_value[window_length:].isin([speak_value])) and array_value[array_value == speak_value].count() >= 10:
        # ウィンドウ内に発話ラベルが1つ以上、30つ以内であれば1、それ以外0
        return 1
    else:
        return 0


def judge_speaking(array_value):
    # ウィンドウ内に全て発話ラベルであれば1、それ以外0
    if all(array_value.isin([0.0])):
        return 1
    else:
        return 0


def judge_endSpeak(array_value):
    window_length = round(len(array_value)/2)
    # ウィンドウ内の後半部分に1が含まれている
    if any(array_value[window_length:].isin([non_speak_value])) and all(array_value[:window_length].isin([speak_value])) and array_value[array_value == speak_value].count() >= 10:
        # ウィンドウ内に発話ラベルが1つ以上、30つ以内であれば1、それ以外0
        return 1
    else:
        return 0


def count_other_au(array_value):
    # series内列ごとにの0より大きい値をカウント
    if any(array_value.isin([np.nan])):
        return 0

    return (array_value > 0).sum()


def convert_df_to_csv(path, df):
    df.to_csv(
        path,
        mode="w",  # 上書き
        index=True,
    )
