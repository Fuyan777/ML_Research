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
        speak_prediction_time
    ):
        # csvに読み込み
        df_face = pd.read_csv(
            resources.face_feature_csv +
            "/%s-feature/previous-feature-value/pre-feat_val_%s.csv" % (user_charactor,
                                                                        speak_prediction_time),
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

        print("----- START : oad_speck_txt_data -------")

        start_speak = []  # 発話開始時間
        end_speak = []  # 発話終了時間
        speak_label = []  # 発話：x、非発話：s

        f = open(
            "elan_output_txt/%s.txt" % face_data_path,
            "r",
            encoding="UTF-8"
        )
        array_data = []
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

        print("----- FINISH : load_speck_txt_data -------\n")

        return (start_speak, end_speak, speak_label)

        #
        # 特徴量のロード
        #

    def load_feature_value(
        self,
        user_charactor,
        speak_prediction_time,
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
            resources.face_feature_csv + "/%s-feature/feature-value/feat_val_%s.csv"
            % (user_charactor, speak_prediction_time),
            encoding="utf-8",
        )
        speak_data = pd.DataFrame(df, columns=resources.feature_reindex_colums)

        # 各クラスのデータ数 確認
        print("y_pre_label: 0")
        print(len(speak_data[speak_data["y_pre_label"] == 0].index))
        print("y_pre_label: 1")
        print(len(speak_data[speak_data["y_pre_label"] == 1].index))

        # オーバーサンプリング
        speak_0_lim = speak_data[speak_data["y_pre_label"] == 0].head(200)
        speak_1_lim = speak_data[speak_data["y_pre_label"] == 1].head(200)

        # print(speak_0_lim)
        # print(speak_1_lim)

        speak_feature_value = pd.concat([speak_0_lim, speak_1_lim])
        print(speak_feature_value)

        print("----- FINISH : looad_feature_value -------\n")

        return speak_feature_value
