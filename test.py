import pandas as pd

non_speak_value = 1
speak_value = 0


def judge_endSpeak(array_value):
    window_length = round(len(array_value)/2)
    # ウィンドウ内の後半部分に1が含まれている
    if any(array_value[window_length:].isin([non_speak_value])):
        print("success1")
        # ウィンドウ内に発話ラベルが1つ以上、30つ以内であれば1、それ以外0
        if array_value[array_value == speak_value].count() >= 10:
            print("success2 end")
            return 1
        else:
            print("failure2")
            return 0
    else:
        print("failure1")
        return 0


# 1,1,1,1,1,
# 0,0,0,0,0,
list = pd.Series([0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0,
                  1, 1, 1])

result = judge_endSpeak(list)
print(result)
