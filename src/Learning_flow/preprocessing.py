from learning_flow import dataset


class Preprocessing:
    def __init__(self):
        print("Preprocessing")

    def extraction_speak_features(self):
        print("extraction_speak_features")
        data = dataset.Dataset()

        # 発話（.txt）データのロード
        # TODO: 後ほどArray<String>に変更
        (start_speak,
            end_speak,
            speak_label) = data.load_speck_txt_data("a-20210128")

        face_feature_data = data.load_face("a-20210128")

        # run
        extraction_speak_by_speak(start_speak, end_speak, speak_label)
        create_csv_labeling_face_by_speak(
            start_speak,
            end_speak,
            speak_label,
            face_feature_data
        )

    #
    # 発話データをもとに顔特徴データにラベリング
    #


def create_csv_labeling_face_by_speak(start_speak, end_speak, speak_label, face_data):
    """ description

    Parameters
    ----------
    start_speak : start speak time.
    end_speak :  end speak time.
    speak_label : speak: x, non-speak：s
    face_data : pandas data from openface output file


    Returns
    ----------
    non(create csv)

    """

    #
    # 発話特性の抽出
    #


def extraction_speak_by_speak(start_speak, end_speak, speak_label):
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

    print("発話回数: %s" % (spk_cnt))
    print("平均発話時間: %s" % (round(speak_interval_average, 3)))
    print("発話最大時間: %s" % (max(speak_interval)))

    print("-------- END : extraction_speak_by_speak ----------\n")
