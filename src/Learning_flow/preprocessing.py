# learning module
from black import user_cache_dir
import matplotlib.collections as collections
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
import os
import seaborn as sns

# SettingWithCopyWarningの非表示
warnings.simplefilter("ignore")

# 発話ラベル
speak_value = 0
# 非発話ラベル
non_speak_value = 1

feature_au_list_forcus = [" AU06_r", " AU12_r", " AU14_r", " AU25_r", " AU26_r"]


class Preprocessing:
    def __init__(self):
        print("Preprocessing")

    def union_all_user_csv_feature_value(self):
        user = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
        data = dataset.Dataset()

        exp_date = 0
        speak_prediction_time = "1w_1s"

        df_total_feature_value = pd.DataFrame()

        for i in range(len(user)):
            if (user[i] == "a") or (user[i] == "b") or (user[i] == "c"):
                exp_date = "20210128"
            elif (user[i] == "d") or (user[i] == "e") or (user[i] == "f"):
                exp_date = "20220106"
            elif (user[i] == "g") or (user[i] == "h") or (user[i] == "i"):
                exp_date = "20220105"

            df = data.load_feature_value(user[i],
                                         speak_prediction_time,
                                         exp_date)

            # df_0 = df[df["y_pre_label"] == 0]
            # df_1 = df[df["y_pre_label"] == 1].head(
            #     len(df[df["y_pre_label"] == 0].index))
            # (df_0)

            # df_sorted = df_0.append(df_1, ignore_index=True)

            # print("df_sported")
            # print(len(df_sorted[df_sorted["y_pre_label"] == 0]))
            # print(len(df_sorted[df_sorted["y_pre_label"] == 1]))
            # print(len(df_sorted.index))

            df_total_feature_value = df_total_feature_value.append(
                df, ignore_index=True)

        print("合計")
        print(df_total_feature_value)
        return df_total_feature_value

    def extraction_speak_features(
        self,
        user_charactor, other1_char, other2_char,
        speak_prediction_time,
        window_size,
        pre_speak_frame,
        user_date,
        exp_date,
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

        (start_speak1, end_speak1, speak_label1) = data.load_speck_txt_other_user_data(exp_date, other1_char)

        (start_speak2, end_speak2, speak_label2) = data.load_speck_txt_other_user_data(exp_date, other2_char)

        # csvから抽出した顔特徴
        df_face_feature = data.load_face(user_date)

        # 認識成功率
        show_recognition_success_rate(df_face_feature)

        # 発話特性データの抽出
        extraction_speak_feature_by_speak(start_speak, end_speak, speak_label)

        # s = df_face_feature[feature_au_list_forcus].corr()
        # # save
        # sns.heatmap(s, square=True, vmax=1,
        #             vmin=-1, center=0, linewidths=.5)
        # plt.savefig(
        #     "/Users/fuyan/LocalDocs/ml-research/ml_graph/heatmap/f_au_heatmap.png")

        # print(speak_label)

        # TODO: 後ほどdf化
        df_my_user_speak = pd.DataFrame(
            data={"label": speak_label,
                  "start_time": start_speak,
                  "end_time": end_speak}
        )
        df_other_1_speak = pd.DataFrame(
            data={"label": speak_label1,
                  "start_time": start_speak1,
                  "end_time": end_speak1}
        )
        df_other_2_speak = pd.DataFrame(
            data={"label": speak_label2,
                  "start_time": start_speak2,
                  "end_time": end_speak2}
        )

        (range_list_my_speak,
         range_list_other_1_speak,
         range_list_other_2_speak) = show_turn_take_visual(df_face_feature,
                                                           df_my_user_speak,
                                                           df_other_1_speak,
                                                           df_other_2_speak,
                                                           user_charactor, exp_date)

        # AUの可視化
        # show_au_time(df_face_feature, speak_label, start_speak,
        #              end_speak, user_charactor, exp_date,
        #              range_list_my_speak, range_list_other_1_speak, range_list_other_2_speak)

        # 顔特徴データのラベリング
        df_feature_reindex = create_csv_labeling_face_by_speak(
            start_speak, end_speak, speak_label,
            start_speak1, end_speak1, speak_label1,
            start_speak2, end_speak2, speak_label2,
            df_face_feature,
            pre_speak_frame,
            user_charactor,
            speak_prediction_time,
            exp_date
        )

        # # マクロ特徴の可視化
        # show_macro_time(df_face_feature, df_feature_reindex, speak_label, start_speak,
        #                 end_speak, user_charactor, exp_date)

        # 可視化のみはここで止める
        # return

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

    # 一から作成する際はこちら
    df_feature_slide = slide.run(window_size, df_face)
    slide_AU = sliding_window.SlidingWindow()
    df_feature_AU_slide = slide_AU.run_AU(window_size, df_face)

    print("--------slide-------")
    print(df_feature_slide)
    print(df_feature_AU_slide)
    print("---------------")

    df_feature_AU_slide.drop(columns=["y", "y_pre_label"], inplace=True)
    print("--------df_feature_slide_dropped-------")
    print(df_feature_AU_slide)

    df_all_data = pd.concat([df_feature_slide, df_feature_AU_slide], axis=1)
    print("--------df_all_data-------")
    print(df_all_data)

    # 修了
    end_time = time.perf_counter()

    # 経過時間を出力(秒)
    elapsed_time = end_time - start_time
    print(elapsed_time)

    # 特徴量をcsvに書き込み

    featues_path = resources.face_feature_csv + "/%s-feature/feature-value/feat_val_%s_%s.csv"
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

    featues_path = resources.face_feature_csv + "/%s-feature/feature-value/feat_val_%s_%s_AU.csv"
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
        ts_df = df_face[(df_face[" timestamp"] >= start_time[index]) & (df_face[" timestamp"] <= end_time[index])]

        # 口の開き具合の算出
        ts_df["mouth"] = ts_df[" y_66"].copy() - ts_df[" y_62"].copy()

        # 視線斜めの算出（hypotenuse）
        ts_df.loc[
            (ts_df[" gaze_angle_x"] < 0.0) | (ts_df[" gaze_angle_y"] < 0.0), "gaze_angle_hypo",
        ] = -(np.sqrt(ts_df[" gaze_angle_x"] ** 2 + ts_df[" gaze_angle_y"] ** 2))

        ts_df.loc[
            ~((ts_df[" gaze_angle_x"] < 0.0) | (ts_df[" gaze_angle_y"] < 0.0)), "gaze_angle_hypo",
        ] = np.sqrt(ts_df[" gaze_angle_x"] ** 2 + ts_df[" gaze_angle_y"] ** 2)

        # 発話 or 非発話　のラベル付け
        ts_df["y"] = speak_value if label[index] == "x" else non_speak_value

        ts_df_feature = ts_df.drop([" timestamp", " y_62", " y_66"], axis=1)

        # 少数第三まで
        df_feature = round(ts_df_feature, 3)

        # 縦に結合
        df_header = pd.concat([df_header, df_feature])

    df_header_other1 = pd.DataFrame()
    for index in range(len(speak_label1)):
        ts_df_other1 = df_face[(df_face[" timestamp"] >= start_time1[index]) & (df_face[" timestamp"] <= end_time1[index])]
        # 0が発話, 1が非発話
        ts_df_other1["isSpeak_other1"] = speak_value if speak_label1[index] == "speech" else non_speak_value
        is_speak_other1 = ts_df_other1["isSpeak_other1"]
        # 縦に結合
        df_header_other1 = pd.concat([df_header_other1, is_speak_other1], axis=0)
    df_header_other1.columns = ["isSpeak_other1"]

    df_header_other2 = pd.DataFrame()
    for index in range(len(speak_label2)):
        ts_df_other2 = df_face[(df_face[" timestamp"] >= start_time2[index]) & (df_face[" timestamp"] <= end_time2[index])]
        ts_df_other2["isSpeak_other2"] = speak_value if speak_label2[index] == "speech" else non_speak_value
        is_speak_other2 = ts_df_other2["isSpeak_other2"]
        # 縦に結合
        df_header_other2 = pd.concat([df_header_other2, is_speak_other2], axis=0)
    df_header_other2.columns = ["isSpeak_other2"]

    # 数秒先をラベリング
    df_header["y_pre_label"] = df_header["y"].shift(-pre_speak_frame)
    df_header["y_pre_label_0.5s"] = df_header["y"].shift(-6)
    df_header["y_pre_label_2s"] = df_header["y"].shift(-24)
    df_header["y_pre_label_3s"] = df_header["y"].shift(-36)
    df_header["y_pre_label_5s"] = df_header["y"].shift(-60)

    df_header_all = pd.DataFrame(df_header, columns=resources.columns_setting_pre_feature_header_all_feature)

    # other1,2をデータに結合
    df_joined = df_header_all.join([df_header_other1, df_header_other2])
    # NaNを見つけたら1(non-speak) を代入
    df_joined[df_joined["isSpeak_other1"].isna() == True] = non_speak_value
    df_joined[df_joined["isSpeak_other2"].isna() == True] = non_speak_value

    # 和集合をとって，他者のどちらか一方が発話しているかのflagを作成
    df_joined["isSpeak_other"] = non_speak_value
    df_joined.loc[(df_joined["isSpeak_other1"] == speak_value), "isSpeak_other"] = speak_value
    df_joined.loc[(df_joined["isSpeak_other2"] == speak_value), "isSpeak_other"] = speak_value

    # 発話時間の代入
    zero = 0
    speak_frame_time = round(df_face[" timestamp"][1], 4)
    df_joined["duration_of_speak_other1"] = zero
    df_joined["duration_of_speak_other2"] = zero
    df_joined["duration_of_speak_other"] = zero

    df_joined.loc[df_joined["isSpeak_other1"] == speak_value, "duration_of_speak_other1"] = speak_frame_time
    df_joined.loc[df_joined["isSpeak_other2"] == speak_value, "duration_of_speak_other2"] = speak_frame_time
    df_joined.loc[df_joined["isSpeak_other"] == speak_value, "duration_of_speak_other"] = speak_frame_time


    # 無音時間の代入
    df_joined["duration_of_non_speak_other1"] = zero
    df_joined["duration_of_non_speak_other2"] = zero
    df_joined["duration_of_non_speak_other"] = zero

    df_joined.loc[df_joined["isSpeak_other1"] == non_speak_value, "duration_of_non_speak_other1"] = speak_frame_time
    df_joined.loc[df_joined["isSpeak_other2"] == non_speak_value, "duration_of_non_speak_other2"] = speak_frame_time
    df_joined.loc[df_joined["isSpeak_other"] == non_speak_value, "duration_of_non_speak_other"] = speak_frame_time

    # 継続時間の積み上げ
    d_s = "duration_of_speak_other"
    d_non_s = "duration_of_non_speak_other"
    # df_joined.loc[d_s]
    # df_joined.loc[d_non_s]

    index_df_joined = len(df_joined)
    # print(df_joined)
    d_s_columns = len(df_joined.columns) - 4
    # print(d_s_columns)
    # print(df_joined.iloc[:,d_s_columns])
    d_non_s_columns = len(df_joined.columns) - 1

    for i in range(index_df_joined):

        if i > 0:
            # 一つ前の行の要素が0以上なら前の要素を加算，0なら何もしない
            if df_joined.iat[i-1, d_s_columns] > 0 and df_joined.iat[i, d_s_columns] > 0:
                df_joined.iat[i, d_s_columns] += df_joined.iat[i-1, d_s_columns]

            if df_joined.iat[i-1, d_non_s_columns] > 0 and df_joined.iat[i, d_non_s_columns] > 0:
                df_joined.iat[i, d_non_s_columns] += df_joined.iat[i-1, d_non_s_columns]

    # NaNの削除
    df_feature = df_joined.dropna()
    print(df_feature)

    show_labeling_data_count(df_feature)

    df_feature_reindex = df_feature.reindex(columns=resources.columns_setting_pre_feature_header_all_feature_re)

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

    return df_feature_reindex


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

#
# 認識成功率の可視化
#


def show_recognition_success_rate(face_feature):
    all_count = len(face_feature[" success"])
    success_count = len(face_feature[face_feature[" success"] == 1])
    failuree_count = len(face_feature[face_feature[" success"] == 0])
    print("【認識成功率】")
    print("成功率　　　: %s" % (success_count / all_count))
    print("失敗率　　　: %s" % (failuree_count / all_count))


feature_macro_list = [" gaze_angle_x", " gaze_angle_y", " pose_Tx", " pose_Ty",
                      " pose_Tz", " pose_Rx", " pose_Ry", " pose_Rz", "mouth"]


def show_macro_time(df_face_feature, df_face_calc, speak_label, speak_start,
                    speak_end, user_charactor, exp_date):
    fontsize = 18
    df_timestamp = df_face_feature[" timestamp"]
    df_mouth = df_face_feature[" y_66"].copy() - df_face_feature[" y_62"].copy()

    # save image
    saving_visual_image_path = "/Users/fuyan/LocalDocs/ml-research/ml_graph/%s_micro_visual" % (user_charactor)

    if not os.path.exists(saving_visual_image_path):
        os.mkdir(saving_visual_image_path)

    # gaze
    fig_mouth, ax_mouth = plt.subplots(figsize=(10, 6))
    ax_mouth.plot(df_timestamp.values, df_mouth, color='orange')
    ax_mouth.set_ylim(0, 100)
    ax_mouth.set_xlim(0, 600)
    ax_mouth.set_xlabel('meeting time [second]', fontsize=fontsize)
    ax_mouth.set_ylabel('output value', fontsize=fontsize)

    # pose
    fig_pose_Tx, ax_pose_Tx = plt.subplots(figsize=(10, 6))
    ax_pose_Tx.plot(df_timestamp.values,
                    df_face_feature[" pose_Tx"], color='orange')
    ax_pose_Tx.set_ylim(0, 100)
    ax_pose_Tx.set_xlim(0, 600)
    ax_pose_Tx.set_xlabel('meeting time [second]', fontsize=fontsize)
    ax_pose_Tx.set_ylabel('output value', fontsize=fontsize)

    # pose
    fig_pose_Tz, ax_pose_Tz = plt.subplots(figsize=(10, 6))
    ax_pose_Tz.plot(df_timestamp.values,
                    df_face_feature[" pose_Tz"], color='orange')
    ax_pose_Tz.set_ylim(300, 600)
    ax_pose_Tz.set_xlim(15, 40)
    ax_pose_Tz.set_xlabel('meeting time [second]', fontsize=fontsize)
    ax_pose_Tz.set_ylabel('output value', fontsize=fontsize)

    # 発話の可視化
    df_speak_info = pd.DataFrame(
        data={"label": speak_label,
              "start_time": speak_start,
              "end_time": speak_end}
    )

    df_speak_label = df_speak_info[df_speak_info["label"] == "x"]
    df_start_time = df_speak_label["start_time"]
    df_end_time = df_speak_label["end_time"]

    # 発話区間の描写
    for index in range(len(df_speak_label)):
        collection = collections.BrokenBarHCollection.span_where(
            df_timestamp.values,
            ymin=300, ymax=600,
            where=(df_timestamp >= df_start_time.iloc[index]) & (
                df_timestamp <= df_end_time.iloc[index]),
            facecolor='blue', alpha=0.5)
        # ax_mouth.add_collection(collection)
        # ax_pose_Tx.add_collection(collection)
        ax_pose_Tz.add_collection(collection)

    fig_mouth.savefig(saving_visual_image_path+"/mouth.png")
    fig_pose_Tx.savefig(saving_visual_image_path+"/pose_Tx.png")
    fig_pose_Tz.savefig(saving_visual_image_path+"/pose_Tz.png")

    plt.show()


#
# AUの時系列可視化
#


# feature_au_list = [" timestamp", " AU01_r", " AU02_r", " AU04_r", " AU05_r", " AU06_r",
#                    " AU07_r", " AU09_r", " AU10_r", " AU12_r", " AU14_r", " AU15_r",
#                    " AU17_r", " AU20_r", " AU23_r", " AU25_r", " AU26_r", " AU45_r"]

# feature_au_arg = [" AU01_r", " AU02_r", " AU04_r", " AU05_r", " AU06_r",
#                   " AU07_r", " AU09_r", " AU10_r", " AU12_r", " AU14_r", " AU15_r",
#                   " AU17_r", " AU20_r", " AU23_r", " AU25_r", " AU26_r", " AU45_r"]

# au_id = ["AU01", "AU02", "AU04", "AU05", "AU06",
#          "AU07", "AU09", "AU10", "AU12", "AU14", "AU15",
#          "AU17", "AU20", "AU23", "AU25", "AU26", "AU45", "i", "g", "h"]

# au_ticks = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19,
#             21, 23, 25, 27, 29, 31, 33, 35, 37, 39]


feature_au_list = [" timestamp", " AU01_r", " AU02_r", " AU04_r", " AU05_r", " AU06_r",
                   " AU07_r", " AU09_r", " AU10_r", " AU12_r", " AU14_r", " AU15_r",
                   " AU17_r", " AU20_r", " AU23_r", " AU25_r", " AU26_r", " AU45_r"]

# b
feature_au_arg = [" AU04_r", " AU06_r", " AU17_r"]

au_id = [" AU04_r", " AU06_r", " AU17_r",  "B", "A", "C"]

# a
# feature_au_arg = [" AU06_r", " AU12_r", " AU25_r"]

# au_id = [" AU06_r", " AU12_r", " AU25_r",  "A", "B", "C"]


au_ticks = [1, 3, 5, 7, 9, 11]

color = ["lightgray", "peachpuff", "orange", "red", "darkred", "black"]


def show_au_time(df, label, start_time, end_time, user_charactor, exp_date, speak_list, speak_list1, speak_list2):
    fontsize = 24

    df_au = df[feature_au_list]
    print("df_au")
    print(df_au)

    # xminの抽出
    print(df_au.at[1, " timestamp"])
    xwidth = df_au.at[1, " timestamp"]

    fig, ax = plt.subplots(figsize=(15, 8))
    plt.rcParams["font.size"] = fontsize

    list_timestamp = df[" timestamp"].values
    # 目盛りの生成処理
    for index_au_id in range(len(feature_au_arg)):
        range_0_list = []  # 0の時
        range_0to1_list = []  # 0以上1未満
        range_1to2_list = []  # 1以上2未満
        range_2to3_list = []  # 2以上3未満
        range_3to4_list = []  # 3以上4未満
        range_4to5_list = []  # 4以上5未満

        # au特徴を配列化
        list_df = df[feature_au_arg[index_au_id]].values

        for i in range(len(df)):
            if list_df[i] > 0.0 and list_df[i] < 1.0:
                range_0to1_list.append((list_timestamp[i], xwidth))
            elif list_df[i] > 1.0 and list_df[i] < 2.0:
                range_1to2_list.append((list_timestamp[i], xwidth))
            elif list_df[i] > 2.0 and list_df[i] < 3.0:
                range_2to3_list.append((list_timestamp[i], xwidth))
            elif list_df[i] > 3.0 and list_df[i] < 4.0:
                range_3to4_list.append((list_timestamp[i], xwidth))
            elif list_df[i] > 4.0 and list_df[i] < 5.0:
                range_4to5_list.append((list_timestamp[i], xwidth))
            else:
                range_0_list.append((list_timestamp[i], xwidth))

        make_broken_barh(ax, range_0_list, index_au_id, color[0])
        make_broken_barh(ax, range_0to1_list, index_au_id, color[1])
        make_broken_barh(ax, range_1to2_list, index_au_id, color[2])
        make_broken_barh(ax, range_2to3_list, index_au_id, color[3])
        make_broken_barh(ax, range_3to4_list, index_au_id, color[4])
        make_broken_barh(ax, range_4to5_list, index_au_id, color[5])

        ax.broken_barh(speak_list,
                       (6, 2), facecolors="blue")
        ax.broken_barh(speak_list1,
                       (8, 2), facecolors="purple")
        ax.broken_barh(speak_list2,
                       (10, 2), facecolors="green")

    # 値の範囲
    ax.set_ylim(0, 12)
    ax.set_xlim(0, 690)
    ax.set_xlabel('meeting time [second]', fontsize=fontsize)
    ax.set_ylabel('Action Unit / Speak Status', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    # yの目盛り軸
    ax.set_yticks(au_ticks)
    ax.set_yticklabels(au_id, fontsize=fontsize)

    for i in range(len(au_ticks)):
        ax.axhline(au_ticks[i] + 1, linewidth=0.5, color="black")

    # 区切り線
    ax.axhline(6, linewidth=3.0, color="black")

    red_patch = mpatches.Patch(color=color[0], label='0')
    red_patch1 = mpatches.Patch(color=color[1], label='0.1-0.9')
    red_patch2 = mpatches.Patch(color=color[2], label='1.0-1.9')
    red_patch3 = mpatches.Patch(color=color[3], label='2.0-2.9')
    red_patch4 = mpatches.Patch(color=color[4], label='3.0-3.9')
    red_patch5 = mpatches.Patch(color=color[5], label='4.0-5.0')

    ax.legend(handles=[red_patch, red_patch1, red_patch2, red_patch3, red_patch4, red_patch5], loc='upper left', bbox_to_anchor=(1, 1))

    # 発話の可視化
    df_speak_info = pd.DataFrame(
        data={"label": label,
              "start_time": start_time,
              "end_time": end_time}
    )

    df_speak_label = df_speak_info[df_speak_info["label"] == "x"]
    df_start_time = df_speak_label["start_time"]
    df_end_time = df_speak_label["end_time"]
    plt.tight_layout()

    # 発話区間の描写
    # for index in range(len(df_speak_label)):
    #     collection = collections.BrokenBarHCollection.span_where(
    #         list_timestamp,
    #         ymin=0, ymax=6,
    #         where=(list_timestamp >= df_start_time.iloc[index]) & (
    #             list_timestamp <= df_end_time.iloc[index]),
    #         facecolor='blue', alpha=0.5)
    #     ax.add_collection(collection)

    # save image
    saving_visual_image_path = "/Users/fuyan/LocalDocs/ml-research/ml_graph/time_au_speak_visual/%s_user" % (user_charactor)

    if not os.path.exists(saving_visual_image_path):
        os.mkdir(saving_visual_image_path)

    plt.savefig("/Users/fuyan/LocalDocs/ml-research/ml_graph/time_au_speak_visual/%s_user/%s-%s.png" % (user_charactor, user_charactor, exp_date))
    print("end image")


def make_broken_barh(ax, range_list, index, color):
    ax.broken_barh(range_list, (2*index, 2), facecolors=(color))


#
# 発話状態の可視化
# ※ 出力時は、user_charaをアルファベットの順に並べる必要がある

def show_turn_take_visual(df_face_feature,
                          df_my_user_speak,
                          df_other_1_speak,
                          df_other_2_speak,
                          user_charactor, exp_date):

    # 基準のタイムスタンプ
    list_timestamp = df_face_feature[" timestamp"].values
    fig, ax_speak = plt.subplots(figsize=(10, 7))

    range_list_my_speak = []
    range_list_other_1_speak = []
    range_list_other_2_speak = []
    df_my_speak_x = df_my_user_speak[df_my_user_speak["label"] == "x"]
    df_other_1_speak_x = df_other_1_speak[df_other_1_speak["label"] == "speech"]
    df_other_2_speak_x = df_other_2_speak[df_other_2_speak["label"] == "speech"]

    # calc
    interval_my_speak = df_my_speak_x["end_time"] - df_my_speak_x["start_time"]
    interval_other_1_speak = df_other_1_speak_x["end_time"] - \
        df_other_1_speak_x["start_time"]
    interval_other_2_speak = df_other_2_speak_x["end_time"] - \
        df_other_2_speak_x["start_time"]

    # 発話割合の算出
    show_speak_rate(interval_my_speak, interval_other_1_speak, interval_other_2_speak)

    # my
    interval_my_array = interval_my_speak.values
    start_time_array = df_my_speak_x["start_time"].values

    # other_1
    interval_other_1_array = interval_other_1_speak.values
    start_time_other_1_array = df_other_1_speak_x["start_time"].values

    # other_2
    interval_other_2_array = interval_other_2_speak.values
    start_time_other_2_array = df_other_2_speak_x["start_time"].values

    for index in range(len(df_my_speak_x)):
        range_list_my_speak.append((start_time_array[index], round(interval_my_array[index], 3)))

    for index in range(len(df_other_1_speak_x)):
        range_list_other_1_speak.append((start_time_other_1_array[index], round(interval_other_1_array[index], 3)))

    for index in range(len(df_other_2_speak_x)):
        range_list_other_2_speak.append((start_time_other_2_array[index], round(interval_other_2_array[index], 3)))

    ax_speak.broken_barh(range_list_my_speak, (22.5, 5), facecolors="blue")
    ax_speak.broken_barh(range_list_other_1_speak, (15, 5), facecolors="orange")
    ax_speak.broken_barh(range_list_other_2_speak, (7.5, 5), facecolors="green")

    # 値の範囲
    ax_speak.set_ylim(5, 30)
    ax_speak.set_xlim(0, 690)
    ax_speak.set_xlabel('meeting time [second]', fontsize=16)
    ax_speak.set_ylabel('user', fontsize=16)

    group_id = ""
    user_other_1 = ""
    user_other_2 = ""
    if user_charactor == "a":
        group_id = "g1_" + exp_date
        user_other_1 = "b"
        user_other_2 = "c"
    elif user_charactor == "d":
        group_id = "g2_" + exp_date
        user_other_1 = "e"
        user_other_2 = "f"
    elif user_charactor == "g":
        group_id = "g3_" + exp_date
        user_other_1 = "h"
        user_other_2 = "i"

    ax_speak.set_yticks([10, 17.5, 25])
    ax_speak.set_yticklabels([user_other_2, user_other_1, user_charactor])

    plt.tick_params(labelsize=20)
    plt.tight_layout()

    plt.savefig("/Users/fuyan/LocalDocs/ml-research/ml_graph/time_speak_visual/%s.png" % (group_id))
    print("end image")

    return range_list_my_speak, range_list_other_1_speak, range_list_other_2_speak


def show_speak_rate(speaking_time, speaking_time_other1, speaking_time_other2):
    # total of user speaking time
    speaking_time_total = speaking_time.sum()
    speaking_time_other1_total = speaking_time_other1.sum()
    speaking_time_other2_total = speaking_time_other2.sum()

    # total of all user speaking time
    speak_total = speaking_time_total + speaking_time_other1_total + speaking_time_other2_total

    # calc rate
    speak_rate_my = (speaking_time_total / speak_total)*100
    speak_rate_other1 = (speaking_time_other1_total / speak_total)*100
    speak_rate_other2 = (speaking_time_other2_total / speak_total)*100

    # speak count
    speak_my_cnt = len(speaking_time)
    speak_other1_cnt = len(speaking_time_other1)
    speak_other2_cnt = len(speaking_time_other2)

    # average
    speaking_time_ave = speaking_time.mean()
    speaking_time_other1_ave = speaking_time_other1.mean()
    speaking_time_other2_ave = speaking_time_other2.mean()

    # array
    speak_cnt_array = [speak_my_cnt, speak_other1_cnt, speak_other2_cnt]
    rate_array = [speak_rate_my, speak_rate_other1, speak_rate_other2]
    speak_average = [speaking_time_ave, speaking_time_other1_ave, speaking_time_other2_ave]
    speak_time_array = [speaking_time, speaking_time_other1, speaking_time_other2]

    for i in range(3):
        print("【発話特性データ】")
        print("発話回数, 発話割合, 発話平均時間, 発話最大時間")
        print("{},{},{},{}".format(round(speak_cnt_array[i]), round(rate_array[i], 1), round(
            speak_average[i], 3), round(max(speak_time_array[i]), 3)))
