import re
from resources import resources
import pandas as pd


class Dataset:
    def __init__(self):
        print("Dataset")

    def load_face(self, face_data_path):
        """

        Parameters
        ----------
        face_data_path : speak data path for elan_output_txt


        Returns
        ----------
        face_feature_data : pandas data

        """

        print("-------- START : load_face ----------")

        # ローカルのopen faceで出力されたcsvファイルのパスを指定して、pandasに出力
        face_feature_data = pd.read_csv(
            resources.face_feature_path + face_data_path + ".csv"
        )

        print(face_feature_data)

        print("-------- END : load_face ----------\n")

        return face_feature_data

    #
    # ウィンドウ処理前の顔特徴データのローディング
    #

    def load_previous_window_face_data(
        self,
        user_charactor,
        speak_prediction_time,
        exp_date
    ):
        # csvに読み込み
        df_face = pd.read_csv(
            resources.face_feature_csv +
            "/%s-feature/previous-feature-value/pre-feat_val_%s_%s.csv" % (user_charactor,
                                                                           speak_prediction_time,
                                                                           exp_date),
        )
        return df_face

    #
    # ウィンドウ処理前の顔特徴データのローディング
    #

    def load_previous_window_face_data_AU(
        self,
        user_charactor,
        speak_prediction_time,
        exp_date
    ):
        # csvに読み込み
        df_face = pd.read_csv(
            resources.face_feature_csv +
            "/%s-feature/previous-feature-value/pre-feat_val_%s_%s_AU.csv" % (user_charactor,
                                                                              speak_prediction_time,
                                                                              exp_date),
        )
        return df_face

    #
    # 発話データのローディング
    #

    def load_speck_txt_data(self, face_data_path):
        """ description

        Parameters
        ----------
        face_data_path : speak data path for elan_output_txt


        Returns
        ----------
        start_speak : start speak time.
        end_speak   :  end speak time.
        speak_label : speak: x, non-speak：s.

        """

        print("----- START : add_speck_txt_data -------")

        start_speak = []  # 発話開始時間
        end_speak = []  # 発話終了時間
        speak_label = []  # 発話：x、非発話：s

        f = open(
            "elan_output_txt/%s.txt" % face_data_path,
            "r",
            encoding="UTF-8"
        )

        array_data = []  # 処理用
        datalines = f.readlines()
        # txt内のデータを1行ずつ読み込み、単語ごとに区切っている
        for data in datalines:
            array_data.append(data.split())

        f.close()

        for speak in array_data:
            start_speak.append(float(speak[0]))
            end_speak.append(float(speak[1]))
            speak_label.append(speak[2])

        # error handling
        if speak_label != "":
            print(speak_label)
        else:
            print("Error speak data nothing")
            return

        replace_list = []

        if "x" not in speak_label:
            for s in speak_label:
                if "speech" in s:
                    text = s.replace("speech", "x")
                    replace_list.append(text)
                elif "noise" in s:
                    text = s.replace("noise", "s")
                    replace_list.append(text)
                else:
                    text = s.replace("noEnergy", "s")
                    replace_list.append(text)
        print(replace_list)

        print("----- FINISH : load_speck_txt_data -------\n")

        if not replace_list:
            return (start_speak, end_speak, speak_label)

        return (start_speak, end_speak, replace_list)

    #
    # 発話データのローディング（他者用）
    #
    def load_speck_txt_other_user_data(self, exp_date, user):
        start_speak = []  # 発話開始時間
        end_speak = []  # 発話終了時間
        # 要素はアルファベット順で並べる
        speak_label1 = []  # 他者1の発話状態　発話：x、非発話：s

        f = open(
            "elan_output_txt/%s-%s.txt" % (user, exp_date),
            "r",
            encoding="UTF-8"
        )

        array_data = []  # 処理用
        datalines = f.readlines()
        # txt内のデータを1行ずつ読み込み、単語ごとに区切っている
        for data in datalines:
            array_data.append(data.split())

        f.close()

        for speak in array_data:
            start_speak.append(float(speak[0]))
            end_speak.append(float(speak[1]))
            speak_label1.append(speak[2])

        return (start_speak, end_speak, speak_label1)

    def load_speck_csv_data(self, face_data_path):
        """ description

        Parameters
        ----------
        face_data_path : speak data path for elan_output_txt


        Returns
        ----------
        start_speak : start speak time.
        end_speak   :  end speak time.
        speak_label : speak: x, non-speak：s.

        """

        print("----- START : oad_speck_txt_data -------")

        print("----- FINISH : load_speck_txt_data -------\n")

        #
        # 特徴量のロード
        #

    def load_feature_value(
        self,
        user_charactor,
        speak_prediction_time,
        exp_date,
        header_index
    ):
        """ description

        Parameters
        ----------
        user_charactor
        speak_prediction_time 

        Returns
        ----------

        """

        print("----- START : looad_feature_value --------")

        df = pd.read_csv(
            resources.face_feature_csv + "/%s-feature/feature-value/feat_val_%s_%s.csv"
            % (user_charactor, speak_prediction_time, exp_date),
            encoding="utf-8",
            header=header_index
        )
        speak_data = pd.DataFrame(
            df, columns=resources.feature_all_reindex_colums)

        # 各クラスのデータ数 確認
        print("y_pre_label: 0")
        print(len(speak_data[speak_data["y_pre_label"] == 0].index))
        print("y_pre_label: 1")
        print(len(speak_data[speak_data["y_pre_label"] == 1].index))

        # データをソートしてしまうと時系列処理できなくなるため、一旦そのままを返す
        return speak_data

        #
        # 特徴量のロード（AUのみ）
        #

    def load_feature_value_AU(
        self,
        user_charactor,
        speak_prediction_time,
        exp_date
    ):
        """ description

        Parameters
        ----------
        user_charactor
        speak_prediction_time 

        Returns
        ----------

        """

        print("----- START : looad_feature_value --------")

        df = pd.read_csv(
            resources.face_feature_csv + "/%s-feature/feature-value/feat_val_%s_%s_AU.csv"
            % (user_charactor, speak_prediction_time, exp_date),
            encoding="utf-8"
        )
        speak_data = pd.DataFrame(
            df, columns=resources.feature_colums_reindex_AU
        )

        # 各クラスのデータ数 確認
        print("y_pre_label: 0")
        print(len(speak_data[speak_data["y_pre_label"] == 0].index))
        print("y_pre_label: 1")
        print(len(speak_data[speak_data["y_pre_label"] == 1].index))

        return speak_data

    def load_feature_value_all(
        self,
        user_charactor,
        speak_prediction_time,
        exp_date,
        colums
    ):
        """ description

        Parameters
        ----------
        user_charactor
        speak_prediction_time 

        Returns
        ----------

        """

        print("----- START : looad_feature_value --------")

        df = pd.read_csv(
            resources.face_feature_csv + "/%s-feature/feature-value/feat_val_%s_%s_all.csv"
            % (user_charactor, speak_prediction_time, exp_date),
            encoding="utf-8",
        )
        speak_data = pd.DataFrame(
            df, columns=colums)

        # 各クラスのデータ数 確認
        print("y_pre_label: 0")
        print(len(speak_data[speak_data["y_pre_label"] == 0].index))
        print("y_pre_label: 1")
        print(len(speak_data[speak_data["y_pre_label"] == 1].index))

        # データをソートしてしまうと時系列処理できなくなるため、一旦そのままを返す
        return speak_data
