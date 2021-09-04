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

        print("----- FINISH : oad_speck_txt_data -------\n")

        return (start_speak, end_speak, speak_label)
