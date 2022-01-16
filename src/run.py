from learning_flow import model_selection
from learning_flow import model_building
from learning_flow import preprocessing
# from data_collection import data_collection

# parameter
# ["d", "e", "f"]  # ["a", "b", "c"] ["g", "h", "i"]
user_charactor = ["f"]
other1_char = "d"
other2_char = "e"
speak_prediction_time = ["1w_1s"]

# ウィンドウサイズの設定w（0.033 : 0.5秒先=15, 1秒=30, 2秒=60, 3秒=90, 5秒=150 ）
window_size_normal = [15, 30, 60, 90, 150]
# ウィンドウサイズの設定w（0.083 : 0.5秒先=6, 1秒=12, 2秒=24, 3秒=36, 5秒=60 ）
window_size_big = [6, 12, 24, 36, 60]

# 予測フレームシフトの設定s（ 0.5秒先=6, 1秒=12, 2秒=24, 3秒=36, 5秒=60 ）
pre_speak_frame = [6, 12, 24, 36, 60]
# all: 全て含めたデータ
# abc: 20210128, ghi: 20210615
exp_date = ["20220106"]


def main():
    print("out_predict_speak")

    pre = preprocessing.Preprocessing()

    for user_index in user_charactor:
        window_size = window_size_big if user_index == "a" else window_size_normal

        for date in exp_date:
            user_date = user_index + "-" + date

            # pre.extraction_speak_features(
            #     user_index, other1_char, other2_char,
            #     speak_prediction_time[0],
            #     window_size[1],
            #     pre_speak_frame[1],
            #     user_date,
            #     exp_date[0]
            # )

            # 特徴量 操作
            # mb = model_building.ModelBuilding()
            # mb.set_building_model(
            #     user_index, speak_prediction_time[0], exp_date[0]
            # )

            m = model_selection.ModelSelection()
            m.set_machine_learning_model(
                user_index, speak_prediction_time[0], exp_date[0]
            )


def main_data_collection():
    d = data_collection.Data_collection()
    d.extraction_speak()


if __name__ == "__main__":
    main()
    # main_data_collection()
