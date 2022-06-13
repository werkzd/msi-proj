from operator import index
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import ttest_rel
from tabulate import tabulate

datasets = ['appendicitis', 'balance', 'banana','ecoli4', 'ecoli-0-1_vs_5', 
            'ecoli-0-1_vs_2-3-5', 'ecoli-0-1-4-6_vs_5', 'ecoli-0-2-3-4_vs_5', 'ecoli-0-3-4_vs_5', 'ecoli-0-3-4-7_vs_5-6',
            'ecoli-0-4-6_vs_5', 'ecoli-0-6-7_vs_5', 'magic', 'page-blocks-1-3_vs_4', 'phoneme',
            'vowel0', 'yeast3', 'yeast4', 'yeast5', 'yeast6']

n_datasets = len(datasets)

for data_id, dataset in enumerate(datasets):
    dataset = np.genfromtxt("%s.csv" % (dataset), delimiter=",")
    X = dataset[:, :-1]
    y = dataset[:, -1].astype(int)

    print("\nDataset: " + str(data_id))
    
    clfs = {
    'gnb': GaussianNB(),
    'rfc': RandomForestClassifier(),
    'dtc': DecisionTreeClassifier(random_state=1)
    }

    for clf_id, clf_name in enumerate(clfs):
        print("\nClassifier: " + clf_name)

        classifier = clfs[clf_name]

        metrics = {
            'recall': None,
            'precision': None,
            'specificity': None,
            'f1': None,
            'g-mean': None,
            'bac': None
        }

        allScores = np.load("results"+str(data_id)+clf_name+".npy")

        for metric_id, metric in enumerate(metrics):
        
            combos = {
            'Bagging - None': None,
            'Bagging - ADASYN': None,
            'AdaBoost - None': None,
            'AdaBoost - ADASYN': None
            }
            
            print("\nMetric: " + metric)

            scores = allScores[:,:,metric_id]

            mean = np.mean(scores, axis=1)
            std = np.std(scores, axis=1)   

            for com_id, com_name in enumerate(combos):
                print("\n%s: %.3f (%.2f)" % (com_name, mean[com_id], std[com_id]))

            print("\nFolds:\n", scores)
            
            alfa = .05
            t_statistic = np.zeros((len(combos), len(combos)))
            p_value = np.zeros((len(combos), len(combos)))

            for i in range(len(combos)):
                for j in range(len(combos)):
                    t_statistic[i, j], p_value[i, j] = ttest_rel(scores[i], scores[j])  

            headers = ['Bagging - None', 'Bagging - ADASYN', 'AdaBoost - None', 'AdaBoost - ADASYN']
            names_column = np.array([['Bagging - None'], ['Bagging - ADASYN'], ['AdaBoost - None'], ['AdaBoost - ADASYN']])
            t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
            t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
            p_value_table = np.concatenate((names_column, p_value), axis=1)
            p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
            print("\nt-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table)
            
            advantage = np.zeros((len(combos), len(combos)))
            advantage[t_statistic > 0] = 1
            advantage_table = tabulate(np.concatenate(
                (names_column, advantage), axis=1), headers)
            print("\nAdvantage:\n", advantage_table)

            significance = np.zeros((len(combos), len(combos)))
            significance[p_value <= alfa] = 1
            significance_table = tabulate(np.concatenate(
                (names_column, significance), axis=1), headers)
            print("\nStatistical significance (alpha = 0.05):\n", significance_table)

            stat_better = significance * advantage
            stat_better_table = tabulate(np.concatenate(
                (names_column, stat_better), axis=1), headers)
            print("\nStatistically significantly better:\n", stat_better_table)