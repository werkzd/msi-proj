import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from adasyn import ADASYN
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from strlearn.metrics import recall, precision, specificity, f1_score, geometric_mean_score_1, balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import clone
import matplotlib.pyplot as plt
from math import pi

dataset = 'ecoli-0-1-4-6_vs_5'
dataset = np.genfromtxt("%s.csv" % (dataset), delimiter=",")
X = dataset[:, :-1]
y = dataset[:, -1].astype(int)

#klasyfikatory bazowe
gnb = GaussianNB()
rfc = RandomForestClassifier()
dtc = DecisionTreeClassifier(random_state=1)

#badania Baggingu i AdaBoosta, do odkomentowania w zależności od badanego przypadku
#clf = BaggingClassifier(base_estimator=gnb, n_estimators=5, bootstrap=True, random_state=1)
#clf = BaggingClassifier(base_estimator=rfc, n_estimators=5, bootstrap=True, random_state=1)
clf = BaggingClassifier(base_estimator=dtc, n_estimators=5, bootstrap=True, random_state=1)
#clf = AdaBoostClassifier(base_estimator=gnb, n_estimators=5, random_state=1)
#clf = AdaBoostClassifier(base_estimator=rfc, n_estimators=5, random_state=1)
#clf = AdaBoostClassifier(base_estimator=dtc, n_estimators=5, random_state=1)

#preprocessing, bark albo własna implementacja ADASYN
preprocs = {
    'none': None,
    'ADASYN': ADASYN(random_state=1),
}

metrics = {
    "recall": recall,
    'precision': precision,
    'specificity': specificity,
    'f1': f1_score,
    'g-mean': geometric_mean_score_1,
    'bac': balanced_accuracy_score,
}

n_splits = 5
n_repeats = 2
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1)

scores = np.zeros((len(preprocs), n_splits * n_repeats, len(metrics)))

for fold_id, (train, test) in enumerate(rskf.split(X, y)):
    for preproc_id, preproc in enumerate(preprocs):
        clf = clone(clf)

        if preprocs[preproc] == None:
            X_train, y_train = X[train], y[train]
        else:
            X_train, y_train = preprocs[preproc].fit_resample(
                X[train], y[train])

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X[test])

        for metric_id, metric in enumerate(metrics):
            scores[preproc_id, fold_id, metric_id] = metrics[metric](
                y[test], y_pred)

#zapisanie wynikow
np.save('results', scores)

scores = np.load("results.npy")
scores = np.mean(scores, axis=1).T

#metryki i metody
metrics=["Recall", 'Precision', 'Specificity', 'F1', 'G-mean', 'BAC']
methods=["None", 'ADASYN']
N = scores.shape[0]

#plot
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]
ax = plt.subplot(111, polar=True)
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)
plt.xticks(angles[:-1], metrics)
ax.set_rlabel_position(0)
plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
["0.0","0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9","1.0"],
color="grey", size=7)
plt.ylim(0,1)


for method_id, method in enumerate(methods):
    values=scores[:, method_id].tolist()
    values += values[:1]
    print(values)
    ax.plot(angles, values, linewidth=1, linestyle='solid', label=method)


plt.legend(bbox_to_anchor=(1.15, -0.05), ncol=5)

#zapis i pokazanie wykresu
plt.savefig("radar", dpi=200)
plt.show()