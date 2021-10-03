from learning_flow import model_selection
from learning_flow import preprocessing

# parameter
user_charactor = ["h"]  # ["a", "b", "c"]
speak_prediction_time = ["1w_1s"]

# ウィンドウサイズの設定w（0.033 : 0.5秒先=15, 1秒=30, 2秒=60, 3秒=90, 5秒=150 ）
window_size_normal = [15, 30, 60, 90, 150]
# ウィンドウサイズの設定w（0.083 : 0.5秒先=6, 1秒=12, 2秒=24, 3秒=36, 5秒=60 ）
window_size_big = [6, 12, 24, 36, 60]

# 予測フレームシフトの設定s（ 0.5秒先=6, 1秒=12, 2秒=24, 3秒=36, 5秒=60 ）
pre_speak_frame = [6, 12, 24, 36, 60]
exp_date = ["20210615"]


def main():
    print("out_predict_speak")

    pre = preprocessing.Preprocessing()

    for user_index in user_charactor:
        window_size = window_size_big if user_index == "a" else window_size_normal

        for date in exp_date:
            user_date = user_index + "-" + date

            # pre.extraction_speak_features(
            #     user_index,
            #     speak_prediction_time[0],
            #     window_size[1],
            #     pre_speak_frame[1],
            #     user_date
            # )

            m = model_selection.ModelSelection()
            m.set_machine_learning_model(
                user_index, speak_prediction_time[0])

    return

    m = model_selection.ModelSelection()
    m.set_machine_learning_model(user_charactor[0], speak_prediction_time[0])


if __name__ == "__main__":
    main()
