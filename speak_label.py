from pprint import pprint
import pandas as pd
import file_path
import warnings

# SettingWithCopyWarningの非表示
warnings.simplefilter("ignore")
user = "a"
speak = "speak-pre"


def main():
    start_speak = []
    end_speak = []
    speak_label = []  # 発話：x、非発話：s

    speak_data = extraction_speak_data()

    for speak in speak_data:
        start_speak.append(float(speak[0]))
        end_speak.append(float(speak[1]))
        speak_label.append(speak[2])

    label_face(speak_label, start_speak, end_speak)


#
# 発話・非発話の時間を抽出
#
def extraction_speak_data():
    f = open("elan_output_txt/a-20210128.txt", "r", encoding="UTF-8")
    tmp_data = []
    datalines = f.readlines()

    for data in datalines:
        tmp_data.append(data.split())

    f.close()
    return tmp_data


#
# 顔特徴データのラベリング
#
def label_face(label, start_time, end_time):
    face_feature = pd.read_csv(file_path.face_feature_path + "a-20210128.csv")
    df_face = pd.DataFrame(
        face_feature,
        columns=[
            " timestamp",
            " gaze_angle_x",
            " gaze_angle_y",
            " pose_Tx",
            " pose_Ty",
            " pose_Tz",
            " pose_Rx",
            " pose_Ry",
            " pose_Rz",
            " y_62",
            " y_66",
        ],
    )

    df_header = pd.DataFrame(
        columns=[
            " gaze_angle_x",
            " gaze_angle_y",
            " pose_Tx",
            " pose_Ty",
            " pose_Tz",
            " pose_Rx",
            " pose_Ry",
            " pose_Rz",
            "mouth",
            "y",
        ]
    )

    for index in range(len(label)):
        # 各非発話区間ごとの顔特徴データ
        ts_df = df_face[
            (df_face[" timestamp"] > start_time[index])
            & (df_face[" timestamp"] < end_time[index])
        ]

        # 口の開き具合
        ts_df["mouth"] = ts_df[" y_66"].copy() - ts_df[" y_62"].copy()
        ts_df["y"] = 0 if label[index] == "x" else 1
        ts_df_feature = ts_df.drop([" timestamp", " y_62", " y_66"], axis=1)

        # 少数第三まで
        df_feature = round(ts_df_feature, 3)

        # 縦に結合
        df_header = pd.concat([df_header, df_feature])

    df_header["y_pre_label"] = df_header["y"].shift(60))

    # csvに書き込み
    df_header.to_csv(
        file_path.face_feature_csv + "/%s-feature/feature_value_%s.csv" % (user, speak),
        mode="w",  # 上書き
        # header=False,
        index=False,
    )


# 一旦廃止
# if len(label) == (index + 1):
#     df_header = pd.DataFrame(
#         columns=[
#             " gaze_angle_x",
#             " gaze_angle_y",
#             " pose_Tx",
#             " pose_Ty",
#             " pose_Tz",
#             " pose_Rx",
#             " pose_Ry",
#             " pose_Rz",
#             "mouth",
#             "y",
#         ]
#     )
#     df_feature_graph = pd.concat([df_header, df_feature])
#     df_feature_graph.to_csv(
#         file_path.face_feature_csv
#         + "/%s-feature/feature_value_header_%s.csv" % (user, speak),
#     )


if __name__ == "__main__":
    main()
