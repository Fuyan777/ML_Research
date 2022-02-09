from learning_flow import model_selection
from learning_flow import model_building
from learning_flow import preprocessing
# from data_collection import data_collection

# parameter
# ["a", "b", "c"] ["d", "e", "f"] ["g", "h", "i"]
user_charactor = ["d"]
other1_char = "e"
other2_char = "f"
speak_prediction_time = ["1w_1s"]

# ウィンドウサイズの設定w（0.033 : 0.5秒先=15, 1秒=30, 2秒=60, 3秒=90, 5秒=150 ）
window_size_normal = [15, 30, 60, 90, 150]
# ウィンドウサイズの設定w（0.083 : 0.5秒先=6, 1秒=12, 2秒=24, 3秒=36, 5秒=60 ）
window_size_big = [6, 12, 24, 36, 60]

# 予測フレームシフトの設定s（ 0.5秒先=6, 1秒=12, 2秒=24, 3秒=36, 5秒=60 ）
pre_speak_frame = [6, 12, 24, 36, 60]
# all: 全て含めたデータ
# abc: 20210128, def: 20220106 ghi: 20210615, 20220105
exp_date = ["20220106"]
str_feature_value = "macro+micro/"

# スコア保持用
all_user_recall = []
all_user_precision = []
all_user_f1 = []


def main():
    print("out_predict_speak")

    pre = preprocessing.Preprocessing()

    for user_index in user_charactor:
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

            # 特徴量 操作
            # mb = model_building.ModelBuilding()
            # mb.set_building_model(
            #     user_index, speak_prediction_time[0], exp_date[0]
            # )

            m = model_selection.ModelSelection()
            recall, precision, f1 = m.set_machine_learning_model(
                user_index, str_feature_value, speak_prediction_time[0], exp_date[0]
            )

            all_user_recall.append(recall)
            all_user_precision.append(precision)
            all_user_f1.append(f1)

    # result
    print("final recall score")
    print(all_user_recall[0], all_user_recall[1], all_user_recall[2])

    print("final precision score")
    print(all_user_precision[0], all_user_precision[1], all_user_precision[2])

    print("final f1 score")
    print(all_user_f1[0], all_user_f1[1], all_user_f1[2])


if __name__ == "__main__":
    main()
