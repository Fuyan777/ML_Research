# learning module
from sklearn.model_selection import LeaveOneOut
import lightgbm as lgb
from learning_flow import preprocessing
from learning_flow import dataset
from resources import resources

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

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

    def set_machine_learning_model(self, user_charactor, str_feature_value, speak_prediction_time, exp_date):
        print("learn_random_forest")

        data = dataset.Dataset()
        speak_feature_value = data.load_feature_value(
            user_charactor,
            speak_prediction_time,
            exp_date
        )

        print("\n==================================")
        print(user_charactor)
        print(exp_date)
        print(str_feature_value)
        print("==================================\n")

        # 目的，説明変数の切り分け
        y = speak_feature_value.loc[:, "y_pre_label"]

        print(speak_feature_value)
        X = speak_feature_value.loc[:,
                                    resources.x_variable_feature_all_colums]

        print(y)
        print(X)

        recall, precision, f1 = make_random_forest_k_hold_validation(
            user_charactor,
            str_feature_value,
            X, y,
            exp_date
        )

        return recall, precision, f1

        if user_charactor == "d":
            # holdout
            make_random_forest_model_holdout(X, y)
            return

        make_random_forest_model_timesplit(
            user_charactor,
            X, y,
            exp_date
        )

#
# random forest
# k-fold validation


def make_random_forest_k_hold_validation(
    user_charactor,
    str_feature_value,
    X, y,
    user_date
):
    # 文字列の保持
    output_array = []

    rus = RandomUnderSampler(random_state=0)
    _X, _y = rus.fit_resample(X, y)

    rf = RandomForestClassifier(
        max_depth=3,
        n_estimators=100,
        random_state=0
    ).fit(_X, _y)
    print(_X)
    print(_y)

    print("class 0: {}".format(len(_y[_y == 0])))
    print("all    : {}".format(len(_y)))

    output_array.append("******* " + str_feature_value+" *******\n")
    output_array.append("■ sample")
    output_array.append("class 0: {}".format(len(_y[_y == 0])))
    output_array.append("all    : {}\n".format(len(_y)))

    y_pred = cross_val_predict(rf, _X, _y, cv=5)
    print(confusion_matrix(_y, y_pred))
    cm = confusion_matrix(_y, y_pred)
    tn, fp, fn, tp = cm.flatten()
    output_array.append("■ confusion matrix")
    output_array.append("[ {} {} ]".format(str(tp), str(fn)))
    output_array.append("[ {} {} ]\n".format(str(fp), str(tn)))

    # 精度結果
    scores = cross_val_score(rf, _X, _y, cv=5, scoring="f1")

    # recording
    output_array.append("■ score")
    output_array.append("f_score list: {}".format(scores))
    output_array.append(
        "Average score(F-measure): {}\n".format(round(np.mean(scores), 3)))

    scores_precision = cross_val_score(rf, _X, _y, cv=5, scoring="precision")

    output_array.append("precision list: {}".format(scores_precision))
    output_array.append("Average score(precision): {}\n".format(
        round(np.mean(scores_precision), 3)))

    scores_recall = cross_val_score(rf, _X, _y, cv=5, scoring="recall")

    output_array.append("recall list: {}".format(scores_recall))
    output_array.append(
        "Average score(recall): {}\n".format(round(np.mean(scores_recall), 3)))

    # printing
    print("recall      : {}".format(round(np.mean(scores_recall), 3)))
    print("precision   : {}".format(
        round(np.mean(scores_precision), 3)))
    print("F-measure   : {}".format(round(np.mean(scores), 3)))

    feature = feature_importance(
        _X,
        rf,
        user_charactor,
        user_date
    )
    output_array.append("■ feature importance\n")
    output_array.append("var     importance")

    for i in range(len(feature.values)):
        output_array.append(
            str(feature.values[i][0]) + "   "+str(round(feature.values[i][1], 5)))

    # recording
    write_txt_log_data(
        user_charactor,
        str_feature_value,
        user_date,
        output_array
    )

    # feature_selection_RFE(rf, _X, _y)

    return round(np.mean(scores_recall), 3), round(np.mean(scores_precision), 3), round(np.mean(scores), 3)

#
# recording experiment data
#


def write_txt_log_data(user_caractor, str_feature_value, user_date, value):
    log_path = resources.path + "log/" + str_feature_value

    if not os.path.exists(log_path):
        os.mkdir(log_path)

    f = open(log_path + user_caractor + "_" + user_date + "_log.txt", 'w',
             encoding='UTF-8')

    for i in range(len(value)):
        f.write(value[i] + '\n')

    f.close()

# 学習モデルの構築（random forest）
# 時系列交差検証
#


def make_random_forest_model_timesplit(
    user_charactor,
    X, y,
    user_date
):
    # スコアの保持
    score_array = []
    # 文字列の保持
    output_array = []
    output_array.append("********* evaluation *********")

    # TimeSeriesSplit cross validation
    tscv = TimeSeriesSplit(n_splits=3)
    time_series_cnt = 0

    for train_index, test_index in tscv.split(X):
        print(
            "---------TimeSeriesSplit {}------------".format(time_series_cnt)
        )
        output_array.append("[ " + str(time_series_cnt) + " times]")
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

        rus_train = RandomUnderSampler(random_state=0)
        X_train_resampled, y_train_resampled = rus_train.fit_resample(
            X_train, y_train)

        # test_strategy = {0: train_count_ratio, 1: train_count_ratio}
        rus_test = RandomUnderSampler(random_state=0)
        X_test_resampled, y_test_resampled = rus_test.fit_resample(
            X_test, y_test)

        visal_train_sampled = sorted(Counter(y_train_resampled).items())
        visal_test_sampled = sorted(Counter(y_test_resampled).items())
        print("[Train]\n発話・非発話　: {}\n".format(visal_train_sampled))
        print("[Test]\n発話・非発話 : {}\n".format(visal_test_sampled))

        # データ数の合計
        print("[all]\ntrainデータ数　: %d" % (len(y_train[y_train == 0])*2))
        print("testデータ数　: %d\n" % (len(y_test[y_test == 0])*2))
        output_array.append("■ sample data")
        output_array.append(
            "trainデータ :" + str(len(y_train[y_train == 0])*2))
        output_array.append(
            "testデータ  :" + str(len(y_test[y_test == 0])*2) + "\n")

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

        (tp, tn, fp, fn) = show_confusion_matrix(y_test_resampled, y_pred)
        output_array.append("[" + str(tp)+" " + str(fn) + "]")
        output_array.append("[" + str(fp)+" " + str(tn) + "]\n")

        score_accuracy, score_recall, score_precision, score_f1 = show_predict_score(
            y_test_resampled, y_pred)

        output_array.append("■ times score")
        output_array.append("accuracy  : {}".format(score_accuracy))
        output_array.append("recall    : {}".format(score_recall))
        output_array.append("precision : {}".format(score_precision))
        output_array.append("F-measure : {}\n".format(score_f1))

        print("\naccuracy: {}".format(score_accuracy))
        print("recall: {}".format(score_recall))
        print("precision: {}".format(score_precision))
        print("F-measure: {}\n".format(score_f1))

        score_array.append(score_f1)

        time_series_cnt += 1

    print("f1_score_array")
    print(score_array)
    print("average F-measure")
    print(np.mean(score_array))
    print(np.std(score_array))

    output_array.append("■ final score")
    output_array.append("1_score_array: {}".format(score_array))
    output_array.append(
        "average F-measure: {}".format(round(np.mean(score_array), 3)))
    output_array.append("std: {}\n".format(round(np.std(score_array), 4)))

    # var_top10, importance_top10 = feature_importance(
    feature = feature_importance(
        X_train_resampled,
        rf,
        user_charactor,
        user_date
    )
    output_array.append("■ feature importance\n")
    output_array.append("var     importance")

    for i in range(len(feature.values)):
        output_array.append(
            str(feature.values[i][0]) + "   "+str(feature.values[i][1]))

    # recording
    write_txt_log_data(
        user_charactor,
        user_date,
        output_array
    )


#
# random forest model (holdout)
#


def make_random_forest_model_holdout(X, y):
    print("発話のデータ数")
    print(len(y[y == 0.0]))
    X_train, X_test, y_train, y_test = train_test_split(X, y,                  # 訓練データとテストデータに分割する
                                                        test_size=0.4,       # テストデータの割合
                                                        shuffle=False,        # シャッフルする
                                                        random_state=0)

    rus_train = RandomUnderSampler(random_state=0)
    X_train_resampled, y_train_resampled = rus_train.fit_resample(
        X_train, y_train)

    rf = RandomForestClassifier(
        max_depth=3,
        n_estimators=100,
        random_state=0
    ).fit(X_train_resampled, y_train_resampled)

    # test under sampling
    rus_test = RandomUnderSampler(random_state=0)
    X_test_resampled, y_test_resampled = rus_test.fit_resample(
        X_test, y_test)

    # 精度の可視化
    y_pred = rf.predict(X_test_resampled)
    show_confusion_matrix(y_test_resampled, y_pred)
    show_predict_score(y_test_resampled, y_pred)

    feature_importance(X_train_resampled, rf, "d", "20220106")


def make_random_forest_model_loo(
    user_charactor,
    X, y,
    user_date
):
    make_loo(X, y)


#
# LeaveOneOut
#


def make_loo(X, y):
    loo = LeaveOneOut()
    print(loo.get_n_splits(X))

    # # 説明変数
    # X_train, X_test = [], []
    # # 目的変数
    # y_train, y_test = [], []

    rf = RandomForestClassifier(
        max_depth=3,
        n_estimators=100,
        random_state=0
    )
    rus_train = RandomUnderSampler(random_state=0)
    X_, y_ = rus_train.fit_resample(X, y)
    print(loo.get_n_splits(X_))

    entire_count = loo.get_n_splits(X_)  # テスト回数取得
    correct_answer_count = 0  # 推定が正解だった数初期化
    counter = 0

    for train_index, test_index in loo.split(X_):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X_.iloc[train_index], X_.iloc[test_index]
        y_train, y_test = y_.iloc[train_index], y_.iloc[test_index]
        # print(X_train, X_test, y_train, y_test)

        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        if (y_pred == y_test).all:
            correct_answer_count += 1.0

        counter += 1
        print(counter)

    print("最終結果")
    print(correct_answer_count)
    print(entire_count)
    rate = (float(correct_answer_count) / float(entire_count))  # 正解率を計算
    print(str(rate))
    print(rate)

    # 精度の可視化
    # show_predict_score(y_test_resampled, y_pred)

#
# 学習モデルの構築（random forest）
# entirety of past data 過去データ丸ごと用いた交差検証
#


def make_random_forest_model_past_data(
    user_charactor,
    X, y
):
    data = dataset.Dataset()
    # 分割数
    split = 3
    result_array = []

    # 過去データ抽出用
    speak_feature_value = data.load_feature_value(
        user_charactor,
        "1w_1s",
        "20201015"
    )

    # 過去データの目的，説明変数の切り分け
    y_train_past = speak_feature_value.loc[:, "y_pre_label"]
    X_train_past = speak_feature_value.loc[:,
                                           resources.x_variable_feature_colums_AU]

    # 過去データ学習
    train_rus = RandomUnderSampler(random_state=0)
    X_past_train_resampled, y_past_train_resampled = train_rus.fit_resample(
        X_train_past, y_train_past)

    rf_past = RandomForestClassifier(
        max_depth=3,
        n_estimators=100,
        random_state=0
    ).fit(X_past_train_resampled, y_past_train_resampled)

    # 検証データ
    n = int(len(y) / split)
    split_index = list(split_list(y.index.tolist(), n))

    print("---------------")

    for index in range(split):
        X_test = X.iloc[split_index[index]]
        y_test = y.iloc[split_index[index]]

        test_strategy = {0: 100, 1: 100}
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

    return score_accuracy, score_recall, score_precision, score_f1

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

    return (tp, tn, fp, fn)


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


def feature_importance(X_train, rf, user_charactor, user_date):
    # 変数重要度
    print("\n[Feature Importances]")

    importance = pd.DataFrame(
        {"var": X_train.columns, "importance": rf.feature_importances_}
    )

    importance_top5 = importance.sort_values(
        "importance", ascending=False
    ).head(5)

    print("----importance------")
    print(importance_top5)
    print("--------------------\n")

    # 棒グラフの生成
    show_bar_graph(
        user_charactor,
        importance_top5["importance"].values,
        importance_top5["var"].values,
        user_date
    )

    importance_top10 = importance.sort_values(
        "importance", ascending=False
    ).head(10)

    # ["var"].values, importance_top10["importance"].values
    return importance_top10


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


#
# 学習モデルの構築（lightgbm）
#

def make_lightgbm_model(X_train, y_train, X_test, y_test):
    # データセットを登録
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)

    # LightGBMのハイパーパラメータを設定
    params = {
        "objective": "binary",
        "metric": "binary_logloss"
    }

    # 学習の履歴を入れる入物
    lgb_results = {}

    # lgb_model = lgb.train(params=params,                    # ハイパーパラメータをセット
    #                       train_set=lgb_train,              # 訓練データを訓練用にセット
    #                       valid_sets=[lgb_train, lgb_test],  # 訓練データとテストデータをセット
    #                       valid_names=['Train', 'Test'],    # データセットの名前をそれぞれ設定
    #                       num_boost_round=5,                # 計算回数
    #                       early_stopping_rounds=10,         # アーリーストッピング設定
    #                       evals_result=lgb_results)         # 履歴を保存する

    lgb_model = lgb.LGBMClassifier(max_depth=5)
    lgb_model.fit(X_train, y_train)

    y_pred = lgb_model.predict(X_test)
    y_pred = np.where(y_pred > 0.5, 1, 0)
    print("--------light gbm score-----------")
    show_predict_score(y_test, y_pred)


def show_bar_graph(user_charactor, y, x_label_array, user_date):
    x = [1, 2, 3, 4, 5]

    fig_bar_graph = plt.figure(figsize=(5, 4))
    fig_bar_graph.subplots_adjust(left=0.3)

    plt.title("features importance: {} user".format(user_charactor))
    plt.barh(x, y, align="center")  # 中央寄せで棒グラフ作成
    plt.yticks(x, x_label_array)  # X軸のラベル

    # plt.show()
    fig_bar_graph.savefig(
        resources.path +
        "ml_graph/feature_importance/{}_importance/{}-{}_micro.png".format(
            user_charactor, user_charactor, user_date)
    )


#
# 特徴量選択
#


def feature_selection_RFE(rf, X, y):
    select_features = 15
    n = 5

    print("--------feature_selection_RFE-----------")

    rfe = RFE(
        rf,
        n_features_to_select=select_features,  # 特徴量数の選択
        step=1,
    )
    rfe.fit(X, y)

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

    scores = cross_val_score(rfe_rf, rfeData, y, cv=n, scoring="f1")
    precision = cross_val_score(rfe_rf, rfeData, y, cv=n, scoring="precision")
    recall = cross_val_score(rfe_rf, rfeData, y, cv=n, scoring="recall")

    print("Recall   : {}".format(np.mean(recall)))
    print("Precision: {}".format(np.mean(precision)))
    print("F1 score : {}".format(np.mean(scores)))

    print("-------Finish-----------")


#
# 混同行列表示
#


def make_cm(matrix, columns):
    # matrix numpy配列

    # columns 項目名リスト
    n = len(columns)

    # '正解データ'をn回繰り返すリスト生成
    act = ['正解'] * n
    pred = ['予測'] * n

    # データフレーム生成
    cm = pd.DataFrame(matrix,
                      columns=[pred, columns], index=[act, columns])
    return cm
