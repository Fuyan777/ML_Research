# learning module
import matplotlib.patches as mpatches
from curses import window
from learning_flow import dataset
from learning_flow import sliding_window
from resources import resources

# external module
import pandas as pd
import numpy as np
import warnings
import time
import matplotlib.pyplot as plt


# SettingWithCopyWarningの非表示
warnings.simplefilter("ignore")


class Preprocessing:
    def __init__(self):
        print("Preprocessing")

    def extraction_speak_features(
        self,
        user_charactor, other1_char, other2_char,
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

        (start_speak1, end_speak1, speak_label1) = data.load_speck_txt_other_user_data(
            exp_date, other1_char)

        (start_speak2, end_speak2, speak_label2) = data.load_speck_txt_other_user_data(
            exp_date, other2_char)

        # csvから抽出した顔特徴
        face_feature_data = data.load_face(user_date)

        # 認識成功率
        show_recognition_success_rate(face_feature_data)

        # 発話特性データの抽出
        extraction_speak_feature_by_speak(start_speak, end_speak, speak_label)

        # 顔特徴データのラベリング
        create_csv_labeling_face_by_speak(
            start_speak, end_speak, speak_label,
            start_speak1, end_speak1, speak_label1,
            start_speak2, end_speak2, speak_label2,
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
    slide.count_au_in_window(window_size, df_face)

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
    start_time, end_time, label,
    start_time1, end_time1, speak_label1,  # 他者発話ラベル
    start_time2, end_time2, speak_label2,
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
        # 顔特徴データの対象範囲の指定
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

    df_header_other1 = pd.DataFrame()
    for index in range(len(speak_label1)):
        ts_df_other1 = df_face[
            (df_face[" timestamp"] >= start_time1[index]) &
            (df_face[" timestamp"] <= end_time1[index])
        ]
        ts_df_other1["isSpeak_other1"] = 0 if speak_label1[index] == "speech" else 1
        is_speak_other1 = ts_df_other1["isSpeak_other1"]
        # 縦に結合
        df_header_other1 = pd.concat(
            [df_header_other1, is_speak_other1], axis=0)
    df_header_other1.columns = ["isSpeak_other1"]

    df_header_other2 = pd.DataFrame()
    for index in range(len(speak_label2)):
        ts_df_other2 = df_face[
            (df_face[" timestamp"] >= start_time2[index]) &
            (df_face[" timestamp"] <= end_time2[index])
        ]
        ts_df_other2["isSpeak_other2"] = 0 if speak_label2[index] == "speech" else 1
        is_speak_other2 = ts_df_other2["isSpeak_other2"]
        # 縦に結合
        df_header_other2 = pd.concat(
            [df_header_other2, is_speak_other2], axis=0)
    df_header_other2.columns = ["isSpeak_other2"]

    # 数秒先をラベリング
    df_header["y_pre_label"] = df_header["y"].shift(-pre_speak_frame)

    df_header_all = pd.DataFrame(
        df_header,
        columns=resources.columns_setting_pre_feature_header_all_feature
    )

    # print(df_header_all["y"])
    # show_au_time(df_face)
    # return

    print(df_header_all)
    print(df_header_other2)

    print(df_header_all.join(df_header_other1))
    df_joined = df_header_all.join([df_header_other1, df_header_other2])

    df_joined[df_joined["isSpeak_other1"].isna() == True] = 1
    df_joined[df_joined["isSpeak_other2"].isna() == True] = 1

    # 全ての発話状態を横結合
    # df_header_all = pd.concat([df_header, df_header_other1], axis="columns")

    df_feature = df_joined.dropna()

    show_labeling_data_count(df_feature)

    df_feature_reindex = df_feature.reindex(
        columns=resources.columns_setting_pre_feature_header_all_feature_re
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


def show_au_time(df):
    df_au = df[feature_au_list]
    print("df_au")
    print(df_au)

    # xminの抽出
    print(df_au.at[1, " timestamp"])
    xwidth = df_au.at[1, " timestamp"]

    fig, ax = plt.subplots(figsize=(11, 6))

    # 目盛りの生成処理
    for j in range(len(au_id)):
        range_0_list = []  # 0の時
        range_0to1_list = []  # 0以上1未満
        range_1to2_list = []  # 1以上2未満
        range_2to3_list = []  # 2以上3未満
        range_3to4_list = []  # 3以上4未満
        range_4to5_list = []  # 4以上5未満

        list_df = df[feature_au[j]].values
        list_time = df[" timestamp"].values

        for i in range(len(df)):
            if list_df[i] > 0.0 and list_df[i] < 1.0:
                range_0to1_list.append((list_time[i], xwidth))
            elif list_df[i] > 1.0 and list_df[i] < 2.0:
                range_1to2_list.append((list_time[i], xwidth))
            elif list_df[i] > 2.0 and list_df[i] < 3.0:
                range_2to3_list.append((list_time[i], xwidth))
            elif list_df[i] > 3.0 and list_df[i] < 4.0:
                range_3to4_list.append((list_time[i], xwidth))
            elif list_df[i] > 4.0 and list_df[i] < 5.0:
                range_4to5_list.append((list_time[i], xwidth))
            else:
                range_0_list.append((list_time[i], xwidth))

        make_broken_barh(ax, range_0_list, j, color[0])
        make_broken_barh(ax, range_0to1_list, j, color[1])
        make_broken_barh(ax, range_1to2_list, j, color[2])
        make_broken_barh(ax, range_2to3_list, j, color[3])
        make_broken_barh(ax, range_3to4_list, j, color[4])
        make_broken_barh(ax, range_4to5_list, j, color[5])

    # 値の範囲
    ax.set_ylim(0, 34)
    ax.set_xlim(5, 6)
    ax.set_xlabel('meeting time [second]')
    ax.set_ylabel('Action Unit')
    # yの目盛り軸
    ax.set_yticks(au_ticks)
    ax.set_yticklabels(au_id)
    for i in range(len(au_ticks)):
        ax.axhline(au_ticks[i] + 1, linewidth=0.5, color="black")

    red_patch = mpatches.Patch(color=color[0], label='0')
    red_patch1 = mpatches.Patch(color=color[1], label='0.1-0.9')
    red_patch2 = mpatches.Patch(color=color[2], label='1.0-1.9')
    red_patch3 = mpatches.Patch(color=color[3], label='2.0-2.9')
    red_patch4 = mpatches.Patch(color=color[4], label='3.0-3.9')
    red_patch5 = mpatches.Patch(color=color[5], label='4.0-5.0')

    collection = collections.BrokenBarHCollection.span_where(
        t, ymin=-1, ymax=0, where=s1 < 0, facecolor='red', alpha=0.5)
    ax.add_collection(collection)

    ax.legend(handles=[red_patch, red_patch1, red_patch2,
              red_patch3, red_patch4, red_patch5], loc='upper left', bbox_to_anchor=(1, 1))
    plt.savefig(
        "/Users/fuyan/LocalDocs/ml-research/ml_graph/time_au_visual/h_user/h-20220105_5to6.png")
    print("end image")

    # plt.show()


def make_broken_barh(ax, range_list, index, color):
    ax.broken_barh(range_list, (2*index, 2), facecolors=(color))


feature_au_list = [" timestamp", " AU01_r", " AU02_r", " AU04_r", " AU05_r", " AU06_r",
                   " AU07_r", " AU09_r", " AU10_r", " AU12_r", " AU14_r", " AU15_r",
                   " AU17_r", " AU20_r", " AU23_r", " AU25_r", " AU26_r", " AU45_r"]

feature_au = [" AU01_r", " AU02_r", " AU04_r", " AU05_r", " AU06_r",
              " AU07_r", " AU09_r", " AU10_r", " AU12_r", " AU14_r", " AU15_r",
              " AU17_r", " AU20_r", " AU23_r", " AU25_r", " AU26_r", " AU45_r"]

au_id = ["AU01", "AU02", "AU04", "AU05", "AU06",
         "AU07", "AU09", "AU10", "AU12", "AU14", "AU15",
         "AU17", "AU20", "AU23", "AU25", "AU26", "AU45"]

au_ticks = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19,
            21, 23, 25, 27, 29, 31, 33]

color = ["lightgray", "peachpuff", "orange", "red", "darkred", "black"]
