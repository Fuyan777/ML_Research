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
import csv
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import common
import file_path
from sklearn.feature_selection import RFE


# 被験者の種類
user = "c"

# n-分割
n = 10

# 木の深さ
max_depth = 3

# feature_valueのpath設定
speak = "5w_5s"

df = pd.read_csv(
    "/Users/fuyan/Documents/ml-research/csv/%s-feature/feature_value/feat_val_%s.csv"
    % (user, speak),
    encoding="utf-8",
)
speak_data = pd.DataFrame(df, columns=common.speak_columns)

print(speak_data.index)

y = speak_data.loc[:, "y"]
X = speak_data.loc[:, common.feature_colums_reindex]

x_data = X.loc[:, common.feature_colums_reindex].values
y_data = y.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

rf = RandomForestClassifier(max_depth=max_depth, n_estimators=100, random_state=1000)
rf.fit(X_train, y_train)

y_pred = cross_val_predict(rf, X, y, cv=n)
# y_pred = rf.predict(X_test)

print(confusion_matrix(y, y_pred))

scores = cross_val_score(rf, X, y, cv=n, scoring="f1_macro")
print("f_score : {}\n".format(scores))
print("Average score(F-measure): {}".format(np.mean(scores)))

print("----------")
print("accuracy_score : {}".format(round(accuracy_score(y, y_pred), 3)))
print("recall_score : {}".format(round(recall_score(y, y_pred, average="macro"), 3)))
# print("recall_score : {}".format(round(recall_score(y, y_pred), 3)))
print(
    "precision_score : {}".format(round(precision_score(y, y_pred, average="macro"), 3))
)
# print("precision_score : {}".format(round(precision_score(y, y_pred), 3)))
print("----------")


# 変数重要度
print("Feature Importances:")
fti = rf.feature_importances_

importance = pd.DataFrame(
    {"var": X_train.columns, "importance": rf.feature_importances_}
)

print("----importance------")
print(importance.sort_values("importance", ascending=False).head(10))
print("--------------------")


# tree生成

# estimator = rf.estimators_[0]
# filename = file_path.path + "ml_graph/tree_%s.png" % (speak)
# dot_data = tree.export_graphviz(
#     estimator,
#     out_file=None,
#     filled=True,
#     rounded=True,
#     feature_names=common.feature_colums_reindex,
#     class_names=["speak", "non-speak"],
#     special_characters=True,
# )
# graph = pdp.graph_from_dot_data(dot_data)
# graph.write_png(filename)


# バウンダリ
# ax = plot_decision_regions(x_data, y_data, clf=rf, legend=0)
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles,
#           ['Speak', 'Non-Speak'],
#           framealpha=0.3, scatterpoints=1)

# # plot_decision_regions(x_data, y_data, clf=rf, legend=2,
# #                       filler_feature_values={},
# #                       )

# plt.show()

#
# 相関係数をプロット
#

# vars = ['median_gaze_x', 'median_gaze_y', "median_poze_x", "median_poze_y", "median_poze_z", "median_mouth"]
# pg = sns.pairplot(speak_data, hue='speak')
# pg.savefig('/Users/fuyan/Documents/siraisi_lab/B4/40_program/graph/sns/rf_sns.png')

#
# 相関行列を作成
#

# corr_matrix = speak_data.corr()

# plt.figure()
# sns.heatmap(
#     corr_matrix,
#     square=True,
#     xticklabels=corr_matrix.columns.values,
#     yticklabels=corr_matrix.columns.values,
# )
# plt.savefig("/Users/fuyan/Documents/siraisi_lab/M1/04_program/graph/sns/sns.png")


#
# 特徴量選択
#

rfe = RFE(
    RandomForestClassifier(max_depth=max_depth, n_estimators=100, random_state=1000),
    n_features_to_select=10,  # 特徴量数の選択
    step=1,
)

# 特徴量削減の実行
rfe.fit(X, y)

# 削減実行後のデータを再構成
rfeData = pd.DataFrame(rfe.transform(X), columns=X.columns.values[rfe.support_])

print("Feature ranking by RFF:", rfe.ranking_)
print(rfeData.columns)

X_train, X_test, y_train, y_test = train_test_split(rfeData, y, test_size=0.2)

rf = RandomForestClassifier(max_depth=max_depth, n_estimators=100, random_state=1000)

# 混同行列求める
y_pred = cross_val_predict(rf, X, y, cv=n)

print(confusion_matrix(y, y_pred))

scores = cross_val_score(rf, rfeData, y, cv=n, scoring="f1_macro")

print("Average score: {}".format(np.mean(scores)))
