from re import I
from learning_flow import model_selection
from learning_flow import preprocessing
# from data_collection import data_collection

# parameter
# ["a", "b", "c"] ["d", "e", "f"] ["g", "h", "i"]

user_charactor = ["a"]
other1_char = "b"
other2_char = "c"
speak_prediction_time = ["1w_1s"]

# ウィンドウサイズの設定w（0.033 : 0.5秒先=15, 1秒=30, 2秒=60, 3秒=90, 5秒=150 ）
window_size_normal = [15, 30, 60, 90, 150]
# ウィンドウサイズの設定w（0.083 : 0.5秒先=6, 1秒=12, 2秒=24, 3秒=36, 5秒=60 ）
window_size_big = [6, 12, 24, 36, 60]

# 予測フレームシフトの設定s（ 0.5秒先=6, 1秒=12, 2秒=24, 3秒=36, 5秒=60 ）
pre_speak_frame = [6, 12, 24, 36, 60]
# all: 全て含めたデータ
str_feature_value = "all/"

# abc: 20210128, def: 20220106 ghi: 20210615, 20220105
exp_date = ["20210128"]

# スコア保持用
all_user_recall = []
all_user_precision = []
all_user_f1 = []


def main():
    print("out_predict_speak")

    pre = preprocessing.Preprocessing()

    for user_index in user_charactor:
        exp_date = []

        if (user_index == "a") or (user_index == "b") or (user_index == "c"):
            exp_date.append("20210128")
        elif (user_index == "d") or (user_index == "e") or (user_index == "f"):
            exp_date.append("20220106")
        elif (user_index == "g") or (user_index == "h") or (user_index == "i"):
            exp_date.append("20220105")

        window_size = window_size_big if user_index == "a" else window_size_normal

        for date in exp_date:
            user_date = user_index + "-" + date

            pre.extraction_speak_features(
                user_index, other1_char, other2_char,
                speak_prediction_time[0],
                window_size[1],
                pre_speak_frame[1],
                user_date,
                exp_date[0]
            )
            return

            m = model_selection.ModelSelection()
            recall, precision, f1 = m.set_machine_learning_model(
                user_index, str_feature_value, speak_prediction_time[0], exp_date[0]
            )

            all_user_recall.append(recall)
            all_user_precision.append(precision)
            all_user_f1.append(f1)

    # result
    print("final recall score")
    for i in range(8):
        print(all_user_recall[i], end=",")

    print("\nfinal precision score")
    for i in range(8):
        print(all_user_precision[i], end=",")

    print("\nfinal f1 score")
    for i in range(8):
        print(all_user_f1[i], end=",")


if __name__ == "__main__":
    main()
