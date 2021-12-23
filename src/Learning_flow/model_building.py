# learning module
from learning_flow import preprocessing
from learning_flow import dataset
from resources import resources

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE
from sklearn.model_selection import TimeSeriesSplit
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter


importance_feature_AU_a_space = [
    " AU14_r_max", " AU25_r_min", " AU25_r_max", " AU14_r_min", " AU25_r_std"
]

importance_feature_AU_a = [
    "AU14_r_max", "AU25_r_min", "AU25_r_max", "AU14_r_min", "AU25_r_std"
]

importance_feature_AU_b_space = [
    " AU25_r_std", " AU45_r_std", " AU45_r_max", " AU26_r_max", " AU25_r_max"
]

importance_feature_AU_b = [
    "AU25_r_std", "AU45_r_std", "AU45_r_max", "AU26_r_max", "AU25_r_max"
]

importance_feature_AU_c_space = [
    " AU12_r_max", " AU14_r_max", " AU25_r_max", " AU12_r_std", " AU06_r_max"
]

importance_feature_AU_c = [
    "AU12_r_max", "AU14_r_max", "AU25_r_max", "AU12_r_std", "AU06_r_max"
]

x_colums_a_space = ['y', 'ave_gaze_x', 'std_gaze_x', 'max_gaze_x', 'min_gaze_x', 'med_gaze_x', 'skew_gaze_x', 'kurt_gaze_x', 'ave_gaze_y', 'std_gaze_y', 'max_gaze_y', 'min_gaze_y', 'med_gaze_y', 'skew_gaze_y', 'kurt_gaze_y', 'ave_gaze_hypo', 'std_gaze_hypo', 'max_gaze_hypo', 'min_gaze_hypo', 'med_gaze_hypo', 'skew_gaze_hypo', 'kurt_gaze_hypo', 'ave_pose_Tx', 'std_pose_Tx', 'max_pose_Tx', 'min_pose_Tx', 'med_pose_Tx', 'skew_pose_Tx', 'kurt_pose_Tx', 'ave_pose_Ty', 'std_pose_Ty', 'max_pose_Ty', 'min_pose_Ty', 'med_pose_Ty', 'skew_pose_Ty', 'kurt_pose_Ty', 'ave_pose_Tz', 'std_pose_Tz', 'max_pose_Tz',
                    'min_pose_Tz', 'med_pose_Tz', 'skew_pose_Tz', 'kurt_pose_Tz', 'ave_pose_Rx', 'std_pose_Rx', 'max_pose_Rx', 'min_pose_Rx', 'med_pose_Rx', 'skew_pose_Rx', 'kurt_pose_Rx', 'ave_pose_Ry', 'std_pose_Ry', 'max_pose_Ry', 'min_pose_Ry', 'med_pose_Ry', 'skew_pose_Ry', 'kurt_pose_Ry', 'ave_pose_Rz', 'std_pose_Rz', 'max_pose_Rz', 'min_pose_Rz', 'med_pose_Rz', 'skew_pose_Rz', 'kurt_pose_Rz', 'ave_mouth', 'std_mouth', 'max_mouth', 'min_mouth', 'med_mouth', 'skew_mouth', 'kurt_mouth', " AU14_r_max", " AU25_r_min", " AU25_r_max", " AU14_r_min", " AU25_r_std"]

x_colums_a = ['y', 'ave_gaze_x', 'std_gaze_x', 'max_gaze_x', 'min_gaze_x', 'med_gaze_x', 'skew_gaze_x', 'kurt_gaze_x', 'ave_gaze_y', 'std_gaze_y', 'max_gaze_y', 'min_gaze_y', 'med_gaze_y', 'skew_gaze_y', 'kurt_gaze_y', 'ave_gaze_hypo', 'std_gaze_hypo', 'max_gaze_hypo', 'min_gaze_hypo', 'med_gaze_hypo', 'skew_gaze_hypo', 'kurt_gaze_hypo', 'ave_pose_Tx', 'std_pose_Tx', 'max_pose_Tx', 'min_pose_Tx', 'med_pose_Tx', 'skew_pose_Tx', 'kurt_pose_Tx', 'ave_pose_Ty', 'std_pose_Ty', 'max_pose_Ty', 'min_pose_Ty', 'med_pose_Ty', 'skew_pose_Ty', 'kurt_pose_Ty', 'ave_pose_Tz', 'std_pose_Tz', 'max_pose_Tz',
              'min_pose_Tz', 'med_pose_Tz', 'skew_pose_Tz', 'kurt_pose_Tz', 'ave_pose_Rx', 'std_pose_Rx', 'max_pose_Rx', 'min_pose_Rx', 'med_pose_Rx', 'skew_pose_Rx', 'kurt_pose_Rx', 'ave_pose_Ry', 'std_pose_Ry', 'max_pose_Ry', 'min_pose_Ry', 'med_pose_Ry', 'skew_pose_Ry', 'kurt_pose_Ry', 'ave_pose_Rz', 'std_pose_Rz', 'max_pose_Rz', 'min_pose_Rz', 'med_pose_Rz', 'skew_pose_Rz', 'kurt_pose_Rz', 'ave_mouth', 'std_mouth', 'max_mouth', 'min_mouth', 'med_mouth', 'skew_mouth', 'kurt_mouth', 'AU14_r_max', 'AU25_r_min', 'AU25_r_max', 'AU14_r_min', 'AU25_r_std']


x_colums_b_space = ['y', 'ave_gaze_x', 'std_gaze_x', 'max_gaze_x', 'min_gaze_x', 'med_gaze_x', 'skew_gaze_x', 'kurt_gaze_x', 'ave_gaze_y', 'std_gaze_y', 'max_gaze_y', 'min_gaze_y', 'med_gaze_y', 'skew_gaze_y', 'kurt_gaze_y', 'ave_gaze_hypo', 'std_gaze_hypo', 'max_gaze_hypo', 'min_gaze_hypo', 'med_gaze_hypo', 'skew_gaze_hypo', 'kurt_gaze_hypo', 'ave_pose_Tx', 'std_pose_Tx', 'max_pose_Tx', 'min_pose_Tx', 'med_pose_Tx', 'skew_pose_Tx', 'kurt_pose_Tx', 'ave_pose_Ty', 'std_pose_Ty', 'max_pose_Ty', 'min_pose_Ty', 'med_pose_Ty', 'skew_pose_Ty', 'kurt_pose_Ty', 'ave_pose_Tz', 'std_pose_Tz', 'max_pose_Tz',
                    'min_pose_Tz', 'med_pose_Tz', 'skew_pose_Tz', 'kurt_pose_Tz', 'ave_pose_Rx', 'std_pose_Rx', 'max_pose_Rx', 'min_pose_Rx', 'med_pose_Rx', 'skew_pose_Rx', 'kurt_pose_Rx', 'ave_pose_Ry', 'std_pose_Ry', 'max_pose_Ry', 'min_pose_Ry', 'med_pose_Ry', 'skew_pose_Ry', 'kurt_pose_Ry', 'ave_pose_Rz', 'std_pose_Rz', 'max_pose_Rz', 'min_pose_Rz', 'med_pose_Rz', 'skew_pose_Rz', 'kurt_pose_Rz', 'ave_mouth', 'std_mouth', 'max_mouth', 'min_mouth', 'med_mouth', 'skew_mouth', 'kurt_mouth', " AU25_r_std", " AU45_r_std", " AU45_r_max", " AU26_r_max", " AU25_r_max"]

x_colums_b = ['y', 'ave_gaze_x', 'std_gaze_x', 'max_gaze_x', 'min_gaze_x', 'med_gaze_x', 'skew_gaze_x', 'kurt_gaze_x', 'ave_gaze_y', 'std_gaze_y', 'max_gaze_y', 'min_gaze_y', 'med_gaze_y', 'skew_gaze_y', 'kurt_gaze_y', 'ave_gaze_hypo', 'std_gaze_hypo', 'max_gaze_hypo', 'min_gaze_hypo', 'med_gaze_hypo', 'skew_gaze_hypo', 'kurt_gaze_hypo', 'ave_pose_Tx', 'std_pose_Tx', 'max_pose_Tx', 'min_pose_Tx', 'med_pose_Tx', 'skew_pose_Tx', 'kurt_pose_Tx', 'ave_pose_Ty', 'std_pose_Ty', 'max_pose_Ty', 'min_pose_Ty', 'med_pose_Ty', 'skew_pose_Ty', 'kurt_pose_Ty', 'ave_pose_Tz', 'std_pose_Tz', 'max_pose_Tz',
              'min_pose_Tz', 'med_pose_Tz', 'skew_pose_Tz', 'kurt_pose_Tz', 'ave_pose_Rx', 'std_pose_Rx', 'max_pose_Rx', 'min_pose_Rx', 'med_pose_Rx', 'skew_pose_Rx', 'kurt_pose_Rx', 'ave_pose_Ry', 'std_pose_Ry', 'max_pose_Ry', 'min_pose_Ry', 'med_pose_Ry', 'skew_pose_Ry', 'kurt_pose_Ry', 'ave_pose_Rz', 'std_pose_Rz', 'max_pose_Rz', 'min_pose_Rz', 'med_pose_Rz', 'skew_pose_Rz', 'kurt_pose_Rz', 'ave_mouth', 'std_mouth', 'max_mouth', 'min_mouth', 'med_mouth', 'skew_mouth', 'kurt_mouth', "AU25_r_std", "AU45_r_std", "AU45_r_max", "AU26_r_max", "AU25_r_max"]


x_colums_c_space = ['y', 'ave_gaze_x', 'std_gaze_x', 'max_gaze_x', 'min_gaze_x', 'med_gaze_x', 'skew_gaze_x', 'kurt_gaze_x', 'ave_gaze_y', 'std_gaze_y', 'max_gaze_y', 'min_gaze_y', 'med_gaze_y', 'skew_gaze_y', 'kurt_gaze_y', 'ave_gaze_hypo', 'std_gaze_hypo', 'max_gaze_hypo', 'min_gaze_hypo', 'med_gaze_hypo', 'skew_gaze_hypo', 'kurt_gaze_hypo', 'ave_pose_Tx', 'std_pose_Tx', 'max_pose_Tx', 'min_pose_Tx', 'med_pose_Tx', 'skew_pose_Tx', 'kurt_pose_Tx', 'ave_pose_Ty', 'std_pose_Ty', 'max_pose_Ty', 'min_pose_Ty', 'med_pose_Ty', 'skew_pose_Ty', 'kurt_pose_Ty', 'ave_pose_Tz', 'std_pose_Tz', 'max_pose_Tz',
                    'min_pose_Tz', 'med_pose_Tz', 'skew_pose_Tz', 'kurt_pose_Tz', 'ave_pose_Rx', 'std_pose_Rx', 'max_pose_Rx', 'min_pose_Rx', 'med_pose_Rx', 'skew_pose_Rx', 'kurt_pose_Rx', 'ave_pose_Ry', 'std_pose_Ry', 'max_pose_Ry', 'min_pose_Ry', 'med_pose_Ry', 'skew_pose_Ry', 'kurt_pose_Ry', 'ave_pose_Rz', 'std_pose_Rz', 'max_pose_Rz', 'min_pose_Rz', 'med_pose_Rz', 'skew_pose_Rz', 'kurt_pose_Rz', 'ave_mouth', 'std_mouth', 'max_mouth', 'min_mouth', 'med_mouth', 'skew_mouth', 'kurt_mouth', " AU12_r_max", " AU14_r_max", " AU25_r_max", " AU12_r_std", " AU06_r_max"]

x_colums_c = ['y', 'ave_gaze_x', 'std_gaze_x', 'max_gaze_x', 'min_gaze_x', 'med_gaze_x', 'skew_gaze_x', 'kurt_gaze_x', 'ave_gaze_y', 'std_gaze_y', 'max_gaze_y', 'min_gaze_y', 'med_gaze_y', 'skew_gaze_y', 'kurt_gaze_y', 'ave_gaze_hypo', 'std_gaze_hypo', 'max_gaze_hypo', 'min_gaze_hypo', 'med_gaze_hypo', 'skew_gaze_hypo', 'kurt_gaze_hypo', 'ave_pose_Tx', 'std_pose_Tx', 'max_pose_Tx', 'min_pose_Tx', 'med_pose_Tx', 'skew_pose_Tx', 'kurt_pose_Tx', 'ave_pose_Ty', 'std_pose_Ty', 'max_pose_Ty', 'min_pose_Ty', 'med_pose_Ty', 'skew_pose_Ty', 'kurt_pose_Ty', 'ave_pose_Tz', 'std_pose_Tz', 'max_pose_Tz',
              'min_pose_Tz', 'med_pose_Tz', 'skew_pose_Tz', 'kurt_pose_Tz', 'ave_pose_Rx', 'std_pose_Rx', 'max_pose_Rx', 'min_pose_Rx', 'med_pose_Rx', 'skew_pose_Rx', 'kurt_pose_Rx', 'ave_pose_Ry', 'std_pose_Ry', 'max_pose_Ry', 'min_pose_Ry', 'med_pose_Ry', 'skew_pose_Ry', 'kurt_pose_Ry', 'ave_pose_Rz', 'std_pose_Rz', 'max_pose_Rz', 'min_pose_Rz', 'med_pose_Rz', 'skew_pose_Rz', 'kurt_pose_Rz', 'ave_mouth', 'std_mouth', 'max_mouth', 'min_mouth', 'med_mouth', 'skew_mouth', 'kurt_mouth', "AU12_r_max", "AU14_r_max", "AU25_r_max", "AU12_r_std", "AU06_r_max"]


class ModelBuilding:
    def __init__(self):
        print("Model")

    def set_building_model(self, user_charactor, speak_prediction_time, exp_date):
        print("learn_random_forest")

        data = dataset.Dataset()
        speak_feature_value = data.load_feature_value(
            user_charactor,
            speak_prediction_time,
            exp_date
        )

        print("\n``````````特徴量の結合`````````````")
        print(user_charactor)
        print(exp_date)

        df_feature_value_AU = data.load_feature_value_AU(
            user_charactor,
            speak_prediction_time,
            exp_date
        )

        # AUで重要な特徴量をdfから抽出
        df_au_loc = df_feature_value_AU.loc[:, importance_feature_AU_a_space]
        # AUと顔特徴を結合
        df_X_original_concat = pd.concat(
            [speak_feature_value, df_au_loc], axis=1)
        # 特徴量をcsvに書き込み
        # featues_path = resources.face_feature_csv + \
        #     "/%s-feature/feature-value/feat_val_%s_%s_all.csv"
        # df_X_original_concat.to_csv(
        #     featues_path % (user_charactor, speak_prediction_time, exp_date),
        #     mode="w",  # 上書き
        #     index=False,
        # )

        print("***** COMPLETE CREATE CSV FILE (feat-val) *****")

        # 最新データの目的，説明変数の切り分け
        y_test_past = df_X_original_concat.loc[:, "y_pre_label"]
        X_test_past = df_X_original_concat.loc[:, x_colums_a_space]

        make_random_forest_model_past_data(
            user_charactor,
            X_test_past, y_test_past,
        )


#
# 学習モデルの構築（random forest）
# entirety of past data 過去データ丸ごと用いた交差検証
#

def make_random_forest_model_past_data(
    user_charactor,
    X, y,
):
    data = dataset.Dataset()
    # 分割数
    split = 3
    result_array = []

    # indexの結合
    x_index = resources.feature_reindex_colums
    x_index.extend(importance_feature_AU_a)

    # 過去データ抽出用
    speak_feature_value = data.load_feature_value_all(
        user_charactor,
        "1w_1s",
        "20201015-2",
        x_index
    )

    print(speak_feature_value)

    speak_feature_value_dropped = speak_feature_value.dropna(how='all', axis=1)

    # 過去データの目的，説明変数の切り分け
    y_train_past = speak_feature_value_dropped.loc[:, "y_pre_label"]
    X_train_past = speak_feature_value_dropped.loc[:, x_colums_a]

    # 過去データ学習
    train_rus = RandomUnderSampler(random_state=0)
    X_past_train_resampled, y_past_train_resampled = train_rus.fit_resample(
        X_train_past, y_train_past)

    rf_past = RandomForestClassifier(
        max_depth=3,
        n_estimators=100,
        random_state=0
    ).fit(X_past_train_resampled, y_past_train_resampled)

    select_features = 20
    rfe = RFE(
        rf_past,
        n_features_to_select=select_features,  # 特徴量数の選択
        step=1,
    ).fit(X_past_train_resampled, y_past_train_resampled)

    rfeData = pd.DataFrame(
        rfe.transform(X),
        columns=X.columns.values[rfe.support_]
    )
    print("rfe data")
    print(rfeData)

    # X_train_past_dropped = rfeData.drop("y", axis=1)
    corr_matrix = rfeData.corr()
    print("corr_matrix")
    print(corr_matrix)

    plt.figure(figsize=(9, 8))
    sns.heatmap(corr_matrix,
                square=True,
                xticklabels=corr_matrix.columns.values,
                yticklabels=corr_matrix.columns.values)
    plt.savefig(resources.path +
                "ml_graph/heatmap/{}_corr_matrix.png".format(
                    user_charactor))
    plt.close('all')

    # 検証データ
    n = int(len(y) / split)
    split_index = list(split_list(y.index.tolist(), n))

    print("---------------")

    for index in range(split):
        X_test = X.iloc[split_index[index]]
        y_test = y.iloc[split_index[index]]

        test_strategy = {0: 115, 1: 115}
        test_rus = RandomUnderSampler(
            random_state=0, sampling_strategy=test_strategy)
        X_test_resampled, y_test_resampled = test_rus.fit_resample(
            X_test, y_test)

        print("全体のデータ数")
        print("[Train]\n全体 : {}".format(len(y_past_train_resampled.index)))
        print("[Test]\n全体 : {}\n".format(len(y_test_resampled.index)))

        print("データ割合")
        # 精度の可視化
        visal_train_sampled = sorted(Counter(y_past_train_resampled).items())
        visal_test_sampled = sorted(Counter(y_test_resampled).items())
        print("[Train]\n発話・非発話　: {}\n".format(visal_train_sampled))
        print("[Test]\n発話・非発話 : {}".format(visal_test_sampled))

        y_pred = rf_past.predict(X_test_resampled)
        print("\n[ {} times score ]\n".format(index))

        show_confusion_matrix(y_test_resampled, y_pred)
        show_predict_score(y_test_resampled, y_pred)

        score_f1 = round(f1_score(y_test_resampled, y_pred), 3)
        result_array.append(score_f1)

    print("f1_score_array")
    print(result_array)
    print("平均精度")
    print(np.mean(result_array))
    print(np.std(result_array))
    feature_importance(
        X_past_train_resampled,
        rf_past,
        user_charactor
    )


#
# 予測結果の表示
#


def show_predict_score(y_test, y_pred):
    score_accuracy = round(accuracy_score(y_test, y_pred), 3)
    score_recall = round(recall_score(y_test, y_pred), 3)
    score_precision = round(precision_score(y_test, y_pred), 3)
    score_f1 = round(f1_score(y_test, y_pred), 3)

    print("\naccuracy: {}".format(score_accuracy))
    print("recall: {}".format(score_recall))
    print("precision: {}".format(score_precision))
    print("F1 score: {}\n".format(score_f1))

#
# 混同行列の表示
#


def show_confusion_matrix(y_test, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    print(confusion_matrix(y_test, y_pred).ravel())
    print("tp: {}, tn: {}, fp: {}, fn: {}\n".format(tp, tn, fp, fn))
    print("\n----正しい方---\n")

    confusion_matrix1 = np.array([[tp, fn],
                                  [fp, tn]])
    print(confusion_matrix1)


def split_list(l, n):
    """
    リストをサブリストに分割する
    :param l: リスト
    :param n: サブリストの要素数
    :return:
    """
    for idx in range(0, len(l), n):
        yield l[idx:idx + n]


#
# 変数重要度
#


def feature_importance(X_train, rf, user_charactor):
    # 変数重要度
    print("\n[Feature Importances]")

    importance = pd.DataFrame(
        {"var": X_train.columns, "importance": rf.feature_importances_}
    )

    feature_importance = importance.sort_values(
        "importance", ascending=False
    ).head(5)

    print("----importance------")
    print(feature_importance)
    print("--------------------\n")

    # 棒グラフの生成
    show_bar_graph(
        user_charactor,
        feature_importance["importance"].values,
        feature_importance["var"].values
    )


def show_bar_graph(user_charactor, y, x_label_array):
    x = [1, 2, 3, 4, 5]

    fig_bar_graph = plt.figure(figsize=(5, 4))
    fig_bar_graph.subplots_adjust(left=0.3)

    plt.title("features importance: {} user".format(user_charactor))
    plt.barh(x, y, align="center")  # 中央寄せで棒グラフ作成
    plt.yticks(x, x_label_array)  # X軸のラベル

    # plt.show()
    fig_bar_graph.savefig(
        resources.path +
        "ml_graph/feature_importance/{}_importance_bar_graph_all.png".format(
            user_charactor)
    )
