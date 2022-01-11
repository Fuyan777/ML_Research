# learning module
from learning_flow import dataset
from learning_flow import sliding_window
from resources import resources

# external module
import pandas as pd
import numpy as np
import warnings
import time


# SettingWithCopyWarningの非表示
warnings.simplefilter("ignore")


class Preprocessing:
    def __init__(self):
        print("Preprocessing")

    def extraction_speak_features(
        self,
        user_charactor,
        speak_prediction_time,
        window_size,
        pre_speak_frame,
        user_date,
        exp_date
    ):
        """
        Parameter
        ---------
        user_charactor: String
        speak_prediction_time: String
        window_size: Int
        pre_speak_frame: Int
        user_date: Array<String>

        """

        print("extraction_speak_features")
        data = dataset.Dataset()

        # 発話（.txt）データのロード
        (start_speak,
            end_speak,
            speak_label) = data.load_speck_txt_data(user_date)

        # csvから抽出した顔特徴
        face_feature_data = data.load_face(user_date)

        # 認識成功率
        show_recognition_success_rate(face_feature_data)

        # 発話特性データの抽出
        extraction_speak_feature_by_speak(start_speak, end_speak, speak_label)

        # 顔特徴データのラベリング
        create_csv_labeling_face_by_speak(
            start_speak,
            end_speak,
            speak_label,
            face_feature_data,
            pre_speak_frame,
            user_charactor,
            speak_prediction_time,
            exp_date
        )

        # ウィンドウ処理前の顔特徴データのロード
        previous_window_face_data = data.load_previous_window_face_data(
            user_charactor,
            speak_prediction_time,
            exp_date
        )

        # 特徴量の抽出
        extraction_feature_value(
            previous_window_face_data,
            window_size,
            user_charactor,
            speak_prediction_time,
            exp_date
        )

        return

        # AU
        create_csv_labeling_face_by_speak_AU(
            start_speak,
            end_speak,
            speak_label,
            face_feature_data,
            pre_speak_frame,
            user_charactor,
            speak_prediction_time,
            exp_date
        )

        previous_window_face_data = data.load_previous_window_face_data_AU(
            user_charactor,
            speak_prediction_time,
            exp_date
        )

        extraction_feature_value_AU(
            previous_window_face_data,
            window_size,
            user_charactor,
            speak_prediction_time,
            exp_date
        )


#
# 特徴量の抽出（normal）
#


def extraction_feature_value(
    df_face,
    window_size,
    user_charactor,
    speak_prediction_time,
    exp_date
):
    """ description

    Parameters
    ----------
    df_dace : pandas data of previous window face features data
    window_size : window size
    shift_size : overlap window size
    user_charactor : a, b, c, etc...
    speak_prediction_time : 0.5s, 1.0s, 2.0, etc...


    Returns
    ----------
    non(create feature_value csv)

    """

    print("-------- START : extraction_feature_value ----------")

    # 開始
    start_time = time.perf_counter()

    slide = sliding_window.SlidingWindow()
    df_feature_slide = slide.run(window_size, df_face)
    slide_AU = sliding_window.SlidingWindow()
    df_feature_AU_slide = slide_AU.run_AU(window_size, df_face)

    print("--------slide-------")
    print(df_feature_slide)
    print(df_feature_AU_slide)
    print("---------------")

    df_feature_AU_slide.drop(
        columns=["y", "y_pre_label"], inplace=True
    )
    print("--------df_feature_slide_dropped-------")
    print(df_feature_AU_slide)

    df_all_data = pd.concat(
        [df_feature_slide, df_feature_AU_slide], axis=1
    )
    print("--------df_all_data-------")
    print(df_all_data)

    # 修了
    end_time = time.perf_counter()

    # 経過時間を出力(秒)
    elapsed_time = end_time - start_time
    print(elapsed_time)

    # 特徴量をcsvに書き込み

    featues_path = resources.face_feature_csv + \
        "/%s-feature/feature-value/feat_val_%s_%s.csv"
    df_all_data.to_csv(
        featues_path % (user_charactor, speak_prediction_time, exp_date),
        mode="w",  # 上書き
        index=False,
    )

    print("***** COMPLETE CREATE CSV FILE (feat-val) *****")

    print("-------- END : extraction_feature_value ----------")

#
# 特徴量の抽出（AU）
#


def extraction_feature_value_AU(
    df_face,
    window_size,
    user_charactor,
    speak_prediction_time,
    exp_date
):
    """ description

    Parameters
    ----------
    df_dace : pandas data of previous window face features data
    window_size : window size
    shift_size : overlap window size
    user_charactor : a, b, c, etc...
    speak_prediction_time : 0.5s, 1.0s, 2.0, etc...


    Returns
    ----------
    non(create feature_value csv)

    """

    print("-------- START : extraction_feature_value ----------")

    slide = sliding_window.SlidingWindow()
    df_feature_slide = slide.run(window_size, df_face)

    # 特徴量をcsvに書き込み

    featues_path = resources.face_feature_csv + \
        "/%s-feature/feature-value/feat_val_%s_%s_AU.csv"
    df_feature_slide.to_csv(
        featues_path % (user_charactor, speak_prediction_time, exp_date),
        mode="w",  # 上書き
        index=False,
    )

    print("***** COMPLETE CREATE CSV FILE (feat-val) *****")

    print("-------- END : extraction_feature_value ----------")


#
# 発話データをもとに顔特徴データにラベリング
#


def create_csv_labeling_face_by_speak(
    start_time,
    end_time, label,
    face_data,
    pre_speak_frame,
    user_charactor,
    speak_prediction_time,
    exp_date
):
    """ description

    Parameters
    ----------
    start_speak : start speak time.
    end_speak :  end speak time.
    label : speak: x, non-speak：s
    face_data : pandas data from openface output file
    user_charactor : a, b, c, etc...
    speak_predeiction_time : 0.5s, 1.0s, 2.0, etc...


    Returns
    ----------
    non(create csv)

    """

    print("-------- START : create_csv_labeling_face_by_speak ----------")

    # 誤認識は全て削除
    face_feature_dropped = face_data[face_data[" success"] == 1]
    df_face = pd.DataFrame(
        face_feature_dropped,
        columns=resources.columns_loading_all_feature,
    )

    # train用のheaderにセットしたpandas dataを作成
    df_header = pd.DataFrame(columns=resources.columns_setting_header)

    for index in range(len(label)):
        # 各非発話区間ごとの顔特徴データ
        ts_df = df_face[
            (df_face[" timestamp"] >= start_time[index]) &
            (df_face[" timestamp"] <= end_time[index])
        ]

        # 口の開き具合の算出
        ts_df["mouth"] = ts_df[" y_66"].copy() - ts_df[" y_62"].copy()

        # 視線斜めの算出（hypotenuse）
        ts_df.loc[
            (ts_df[" gaze_angle_x"] < 0.0) | (ts_df[" gaze_angle_y"] < 0.0),
            "gaze_angle_hypo",
        ] = -(np.sqrt(ts_df[" gaze_angle_x"] ** 2 + ts_df[" gaze_angle_y"] ** 2))

        ts_df.loc[
            ~((ts_df[" gaze_angle_x"] < 0.0) | (ts_df[" gaze_angle_y"] < 0.0)),
            "gaze_angle_hypo",
        ] = np.sqrt(ts_df[" gaze_angle_x"] ** 2 + ts_df[" gaze_angle_y"] ** 2)

        # 発話 or 非発話　のラベル付け
        ts_df["y"] = 0 if label[index] == "x" else 1
        ts_df_feature = ts_df.drop([" timestamp", " y_62", " y_66"], axis=1)

        # 少数第三まで
        df_feature = round(ts_df_feature, 3)

        # 縦に結合
        df_header = pd.concat([df_header, df_feature])

    # 数秒先をラベリング
    df_header["y_pre_label"] = df_header["y"].shift(-pre_speak_frame)
    df_feature = df_header.dropna()

    show_labeling_data_count(df_feature)

    df_feature_reindex = df_feature.reindex(
        columns=resources.columns_setting_pre_feature_header_all_feature
    )

    # csvに書き込み
    df_feature_reindex.to_csv(
        resources.face_feature_csv +
        "/%s-feature/previous-feature-value/pre-feat_val_%s_%s.csv" % (user_charactor,
                                                                       speak_prediction_time,
                                                                       exp_date),
        mode="w",  # 上書き
        # header=False,
        index=False,
    )
    print("***** COMPLETE CREATE CSV FILE (pre-feat-val) *****")

    print("-------- END : create_csv_labeling_face_by_speak ----------\n")

#
# 発話データをもとに顔特徴データにラベリング（Action Unit用）
#


def create_csv_labeling_face_by_speak_AU(
    start_time,
    end_time, label,
    face_data,
    pre_speak_frame,
    user_charactor,
    speak_prediction_time,
    exp_date
):
    # 誤認識は全て削除
    face_feature_dropped = face_data[face_data[" success"] == 1]
    df_face = pd.DataFrame(
        face_feature_dropped,
        columns=resources.columns_loading_AU,
    )

    # train用のheaderにセットしたpandas dataを作成
    df_header = pd.DataFrame(columns=resources.columns_setting_header_AU)
    for index in range(len(label)):
        # 各非発話区間ごとの顔特徴データ
        ts_df = df_face[
            (df_face[" timestamp"] >= start_time[index]) &
            (df_face[" timestamp"] <= end_time[index])
        ]
        # 発話 or 非発話　のラベル付け
        ts_df["y"] = 0 if label[index] == "x" else 1
        ts_df_feature = ts_df.drop([" timestamp"], axis=1)

        # 少数第三まで
        df_feature = round(ts_df_feature, 3)

        # 縦に結合
        df_header = pd.concat([df_header, df_feature])

    # 数秒先をラベリング
    df_header["y_pre_label"] = df_header["y"].shift(-pre_speak_frame)
    df_feature = df_header.dropna()
    show_labeling_data_count(df_feature)

    df_feature_reindex = df_feature.reindex(
        columns=resources.columns_setting_pre_feature_header_AU
    )
    # csvに書き込み
    df_feature_reindex.to_csv(
        resources.face_feature_csv +
        "/%s-feature/previous-feature-value/pre-feat_val_%s_%s_AU.csv" % (user_charactor,
                                                                          speak_prediction_time,
                                                                          exp_date),
        mode="w",  # 上書き
        index=False,
    )
    print("***** COMPLETE CREATE CSV FILE (pre-feat-val) *****")

    print("-------- END : create_csv_labeling_face_by_speak ----------\n")


#
# ラベリングデータ表示
#


def show_labeling_data_count(df_feature):
    print("【ウィンドウ処理前の発話ラベリングデータ】")
    print("全発話数　: %s" % (len(df_feature)))
    print("発話数　　: %s" % (len(df_feature[df_feature["y"] == 0])))
    print("非発話数　: %s" % (len(df_feature[df_feature["y"] == 1])))

#
# 発話特性の抽出
#


def extraction_speak_feature_by_speak(start_speak, end_speak, speak_label):
    """ description

    Parameters
    ----------
    start_speak : start speak time.
    end_speak :  end speak time.
    speak_label : speak: x, non-speak：s


    Returns
    ----------
    non

    """

    spk_cnt = 0
    speak_interval = []

    print("-------- START : extraction_speak_by_speak ----------")

    for index in range(len(speak_label)):
        if speak_label[index] == "x":
            spk_cnt += 1
            tmp_speak_interval = end_speak[index] - start_speak[index]
            speak_interval.append(tmp_speak_interval)

    speak_interval_average = sum(speak_interval) / len(speak_interval)

    print("【発話特性データ】")
    print("発話回数　　　: %s" % (spk_cnt))
    print("平均発話時間　: %s" % (round(speak_interval_average, 3)))
    print("発話最大時間　: %s" % (round(max(speak_interval), 3)))

    print("-------- END : extraction_speak_by_speak ----------\n")


def show_recognition_success_rate(face_feature):
    all_count = len(face_feature[" success"])
    success_count = len(face_feature[face_feature[" success"] == 1])
    failuree_count = len(face_feature[face_feature[" success"] == 0])
    print("【認識成功率】")
    print("成功率　　　: %s" % (success_count / all_count))
    print("失敗率　　　: %s" % (failuree_count / all_count))
