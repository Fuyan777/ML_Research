import seaborn as sns
import pydotplus as pdp
from sklearn import tree
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
import common
import file_path
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler

#
# パラメータ設定
#

# 被験者の種類
user = "a"

# n-分割
n = 10

# 木の深さ
max_depth = 3

# feature_valueのpath設定
speak = "1w_1s"

# 特徴量選択の数
select_features = 10


def main():
    # データ読み込み
    df = pd.read_csv(
        "/Users/fuyan/Documents/ml-research/csv/%s-feature-speak-include/feature_value/feat_val_%s.csv"
        % (user, speak),
        encoding="utf-8",
    )
    speak_data = pd.DataFrame(df, columns=common.speak_columns)

    # 各クラスのデータ数 確認
    print("y_pre_label: 0")
    print(len(speak_data[speak_data["y_pre_label"] == 0].index))
    print("y_pre_label: 1")
    print(len(speak_data[speak_data["y_pre_label"] == 1].index))

    # 各クラスのデータ
    speak_0_lim = speak_data[speak_data["y_pre_label"] == 0].head(800)
    speak_1_lim = speak_data[speak_data["y_pre_label"] == 1].head(800)

    # print(speak_0_lim)
    # print(speak_1_lim)

    speak_feature_value = pd.concat([speak_0_lim, speak_1_lim])
    print(speak_feature_value)

    # 標準化処理
    # speak_data = standard_scaler(df_speak)

    # 目的，説明変数の切り分け
    y = speak_feature_value.loc[:, "y_pre_label"]
    X = speak_feature_value.loc[:, common.feature_colums]

    # 決定境界で使用するデータ
    x_data = X.loc[:, common.feature_colums].values
    y_data = y.values

    # 学習データ分割（8:2）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    rf = RandomForestClassifier(
        max_depth=max_depth, n_estimators=100, random_state=1000
    )

    # ランダムフォレストで学習
    predict_random_forest(rf, X_train, y_train, X, y)

    # 特徴量選択
    feature_importance(X, y)

    # 決定境界
    # generate_decision_regions

    # 決定木
    # generate_tree(rf)

    # 相関行列
    generate_heatmap(speak_data)


#
# ランダムフォレストで学習
#


def predict_random_forest(rf, X_train, y_train, X, y):
    rf.fit(X_train, y_train)

    y_pred = cross_val_predict(rf, X, y, cv=n)
    print(confusion_matrix(y, y_pred))

    scores = cross_val_score(rf, X, y, cv=n, scoring="f1_macro")
    print("f_score : {}\n".format(scores))
    print("Average score(F-measure): {}".format(np.mean(scores)))

    print("----------")
    print("accuracy: {}".format(round(accuracy_score(y, y_pred), 3)))
    print("recall: {}".format(round(recall_score(y, y_pred, average="macro"), 3)))
    print("precision: {}".format(
        round(precision_score(y, y_pred, average="macro"), 3)))
    print("----------")

    # 変数重要度
    print("Feature Importances:")

    importance = pd.DataFrame(
        {"var": X_train.columns, "importance": rf.feature_importances_}
    )

    print("----importance------")
    print(importance.sort_values("importance", ascending=False).head(10))
    print("--------------------")


#
# 特徴量選択
#


def feature_importance(X, y):
    rfe = RFE(
        RandomForestClassifier(
            max_depth=max_depth, n_estimators=100, random_state=1000
        ),
        n_features_to_select=select_features,  # 特徴量数の選択
        step=1,
    )

    # 特徴量削減の実行
    rfe.fit(X, y)

    # 削減実行後のデータを再構成
    rfeData = pd.DataFrame(rfe.transform(
        X), columns=X.columns.values[rfe.support_])

    X_train, X_test, y_train, y_test = train_test_split(
        rfeData, y, test_size=0.2)

    rf = RandomForestClassifier(
        max_depth=max_depth, n_estimators=100, random_state=1000
    )

    # 混同行列求める
    y_pred = cross_val_predict(rf, X, y, cv=n)

    print(confusion_matrix(y, y_pred))

    scores = cross_val_score(rf, rfeData, y, cv=n, scoring="f1_macro")

    print("Average score: {}".format(np.mean(scores)))


#
# 決定木の生成
#


def generate_tree(rf):
    estimator = rf.estimators_[3]
    filename = file_path.path + "ml_graph/tree_%s.png" % (speak)
    dot_data = tree.export_graphviz(
        estimator,
        out_file=None,
        filled=True,
        rounded=True,
        feature_names=common.feature_tree,
        class_names=["speak", "non-speak"],
        special_characters=True,
    )
    graph = pdp.graph_from_dot_data(dot_data)
    graph.write_png(filename)


#
# 決定境界図の生成
#


def generate_decision_regions():
    ax = plot_decision_regions(x_data, y_data, clf=rf, legend=0)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, ["Speak", "Non-Speak"], framealpha=0.3, scatterpoints=1)

    plot_decision_regions(
        x_data,
        y_data,
        clf=rf,
        legend=2,
    )

    plt.show()


#
# 相関行列
#


def generate_heatmap(speak_data):
    corr_matrix = speak_data.corr()

    plt.figure()
    sns.heatmap(
        corr_matrix,
        square=True,
        xticklabels=corr_matrix.columns.values,
        yticklabels=corr_matrix.columns.values,
    )
    plt.savefig(
        "/Users/fuyan/Documents/siraisi_lab/M1/04_program/graph/sns/sns.png")


#
# 標準化の処理
#


def standard_scaler(df):
    stdsc = StandardScaler()
    df_std = stdsc.fit_transform(df)
    df_face_std = pd.DataFrame(df_std, columns=common.speak_columns)
    print(df_face_std)
    return df_face_std


if __name__ == "__main__":
    main()
