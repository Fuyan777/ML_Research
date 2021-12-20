# learning module
from matplotlib.pyplot import axis
from pandas.io.formats.format import return_docstring
from resources import resources

# external module
import numpy as np
import pandas as pd

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
        # overlapの計算
        shift_size = (window_size // 2) - 1

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

        print("=========後処理==============")
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
        # dfの結合
        tmp_all_feature = pd.concat(
            [
                df_y,
                df_y_pre,
                df_std,
                df_max,
                df_min
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

# 発話データを除外する処理
# 処理内容 : 配列内に発話が含めば、nanを返す


def judge_y(array_value):
    if any(array_value.isin([np.nan])):
        return np.nan
    elif any(array_value.isin([0.0])):
        return np.nan
    else:  # 発話区間も含む場合、else部分は消す
        return array_value.iloc[-1]  # 配列の最後の要素を返す


def judge_y_pre(array_value):
    return array_value.iloc[-1]
