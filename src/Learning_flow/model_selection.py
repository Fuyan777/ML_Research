# learning module
from learning_flow import preprocessing
from learning_flow import dataset
from resources import resources

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE
from sklearn.model_selection import TimeSeriesSplit
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter


class ModelSelection:
    def __init__(self):
        print("Model")

    def set_machine_learning_model(self, user_charactor, speak_prediction_time):
        print("learn_random_forest")

        data = dataset.Dataset()
        speak_feature_value = data.load_feature_value(
            user_charactor,
            speak_prediction_time
        )

        print("\n==================================")
        print(user_charactor)
        print("==================================\n")

        # 目的，説明変数の切り分け
        y = speak_feature_value.loc[:, "y_pre_label"]
        X = speak_feature_value.loc[:, resources.x_variable_feature_colums]

        print(y)
        print(X)

        # # 学習データ分割
        # X_train, X_test, y_train, y_test = train_test_split(
        #     X,
        #     y,
        #     test_size=0.3,
        #     random_state=0
        # )

        # 過去データの訓練データ構築

        make_random_forest_model_timesplit(
            user_charactor,
            X, y
        )

#
# 学習モデルの構築（random forest）
#


def make_random_forest_model_timesplit(
    user_charactor,
    X, y
):
    # スコアの保持
    score_array = []

    # TimeSeriesSplit cross validation
    tscv = TimeSeriesSplit(n_splits=5)
    time_series_cnt = 0

    for train_index, test_index in tscv.split(X):
        print(
            "---------TimeSeriesSplit {}------------".format(time_series_cnt)
        )
        # 説明変数
        X_train, X_test = [], []
        # 目的変数
        y_train, y_test = [], []

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        print("\nアンダーサンプリング前\n")
        # 訓練データの可視化
        print("[Train]\n発話のデータ数　: %d" % (len(y_train[y_train == 0])))
        print("非発話のデータ数: %d\n" % (len(y_train[y_train == 1])))

        # テストデータの可視化
        print("[Test]\n発話のデータ数　: %d" % (len(y_test[y_test == 0])))
        print("非発話のデータ数: %d" % (len(y_test[y_test == 1])))

        print("\nアンダーサンプリング後")
        rus = RandomUnderSampler(random_state=0)
        X_train_resampled, y_train_resampled = rus.fit_resample(
            X_train, y_train)
        X_test_resampled, y_test_resampled = rus.fit_resample(
            X_test, y_test)

        visal_train_sampled = sorted(Counter(y_train_resampled).items())
        visal_test_sampled = sorted(Counter(y_test_resampled).items())
        print("[Train]\n発話・非発話　: {}\n".format(visal_train_sampled))
        print("[Test]\n発話・非発話 : {}".format(visal_test_sampled))

        # 学習
        rf = RandomForestClassifier(
            max_depth=3,
            n_estimators=100,
            random_state=0
        ).fit(X_train_resampled, y_train_resampled)

        rf_score = rf.score(X_test_resampled, y_test_resampled)

        # 精度の可視化
        y_pred = rf.predict(X_test_resampled)
        print("\n[ {} times score ]\n".format(time_series_cnt))
        print(confusion_matrix(y_test_resampled, y_pred))

        score_accuracy = round(accuracy_score(y_test_resampled, y_pred), 3)
        score_recall = round(recall_score(y_test_resampled, y_pred), 3)
        score_precision = round(precision_score(y_test_resampled, y_pred), 3)
        score_f1 = round(f1_score(y_test_resampled, y_pred), 3)

        print("\naccuracy: {}".format(score_accuracy))
        print("recall: {}".format(score_recall))
        print("precision: {}".format(score_precision))
        print("F1 score: {}\n".format(score_f1))

        score_array.append(score_f1)

        time_series_cnt += 1

    print("f1_score_array")
    print(score_array)
    print("平均精度")
    print(np.mean(score_array))
    print(np.std(score_array))

#
# 変数重要度
#


def feature_importance(X_train, rf, user_charactor):
    # 変数重要度
    print("[Feature Importances]")

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


#
# 学習モデルの構築（random forest）
#


def make_random_forest_model(
        user_charactor,
        X_train, y_train,
        X_test, y_test,
        X, y


):
    max_depth = 3
    n = 10

    # 学習
    rf = RandomForestClassifier(
        max_depth=max_depth,
        n_estimators=100,
        random_state=0
    ).fit(X_train, y_train)

    # k hold cross validation
    y_pred = cross_val_predict(rf, X, y, cv=n)
    print(confusion_matrix(y, y_pred))

    # 精度結果
    scores = cross_val_score(rf, X, y, cv=n, scoring="f1_macro")
    print("f_score list: {}\n".format(scores))
    print("Average score(F-measure): {}".format(np.mean(scores)))

    print("----------")
    print("accuracy: {}".format(round(accuracy_score(y, y_pred), 3)))
    print("recall: {}".format(
        round(recall_score(y, y_pred, average="macro"), 3)))
    print("precision: {}".format(
        round(precision_score(y, y_pred, average="macro"), 3)))
    print("----------")

    # 変数重要度
    print("Feature Importances:")

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

    feature_selection_RFE(rf, X_train, y_train, X, y, n)


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
        "ml_graph/feature_importance/{}_importance_bar_graph.png".format(
            user_charactor)
    )


#
# 特徴量選択
#


def feature_selection_RFE(rf, X_train, y_train, X, y, n):
    select_features = 20

    print("--------feature_selection_RFE-----------")

    rfe = RFE(
        rf,
        n_features_to_select=select_features,  # 特徴量数の選択
        step=1,
    ).fit(X_train, y_train)

    # 削減実行後のデータを再構成
    rfeData = pd.DataFrame(
        rfe.transform(X),
        columns=X.columns.values[rfe.support_]
    )
    rfe_rf = RandomForestClassifier(
        max_depth=3, n_estimators=100, random_state=1000
    )

    y_pred = cross_val_predict(rfe_rf, X, y, cv=n)
    print(confusion_matrix(y, y_pred))

    scores = cross_val_score(rfe_rf, rfeData, y, cv=n, scoring="f1_macro")
    print("Average score: {}".format(np.mean(scores)))

    print("-------Finish-----------")
