import numpy as np
from sklearn.neighbors import NearestNeighbors
from collections import Counter
import numpy as np
#import warnings
#warnings.filterwarnings("ignore")

class ADASYN(object):
    def __init__(self,random_state=None):
        self.beta = 1
        self.d_th = 1
        self.k = 10
        self.random_state = random_state
        self.m_maj = 0
        self.m_min = 0
        self.minor_class = 0

    def degree_of_imbalace(self, X, y):
        d = Counter(y)

        if d[0] > d[1]:
            self.m_maj = d[0]
            self.m_min = d[1]
            self.minor_class = 1
        else:
            self.m_min = d[0]
            self.m_maj = d[1]
            self.minor_class = 0
        d = self.m_min/self.m_maj

        return d

    def num_of_synth_data(self):
        G = (self.m_maj - self.m_min)/self.beta   
        return G

    def calculate_ratio(self, X, y):
        min_x = []
        iterator = X.shape[0]
        for i in range(iterator):
            if y[i] == self.minor_class:
                min_x.append(i)

        self.nn = NearestNeighbors(n_neighbors=self.k + 1).fit(X)
        knn = self.nn.kneighbors(X[min_x], return_distance=False)
        knn_duplicates = y[knn.ravel()].reshape(knn.shape)
        count_knn_duplicates = [Counter(i) for i in knn_duplicates]
        r_i = np.array([(sum(i.values()) - i[1]) / float(self.k) for i in count_knn_duplicates])
        return r_i, min_x, knn, knn_duplicates

    def normalize_ri(self, r_i):
        if np.sum(r_i):
            r_i = r_i / np.sum(r_i)
        return r_i
        
    def calculate_gi(self, G, r_i):
        g_i = np.floor(r_i * G)
        return g_i

    def generate_samples(self, X, y, min_x, knn, count_knn_duplicates, g_i):
        new_X = []
        new_y = []

        for i, e in enumerate(min_x):
            min_knn = [ele for idx, ele in enumerate(knn[i][1:-1]) if count_knn_duplicates[i][idx +1] == self.minor_class]
            if not min_knn:
                continue
            for j in range(0, int(g_i[i])):
                random_i = np.random.random_integers(0, len(min_knn) - 1)
                lambda_var = np.random.random_sample()
                s_i = X[e] + (X[min_knn[random_i]] - X[e]) * lambda_var    
                new_X.append(s_i)
                new_y.append(y[e])

        return(np.asarray(new_X), np.asarray(new_y))

    def adasyn_algorithm(self, X, y):
        d = self.degree_of_imbalace(X, y)

        if d < self.d_th:
            G = self.num_of_synth_data()
            r_i, min_x, knn, knn_duplicates = self.calculate_ratio(X,y)
            r_i = self.normalize_ri(r_i)
            g_i = self.calculate_gi(G, r_i)
            synth_X, synth_y = self.generate_samples(X, y, min_x, knn, knn_duplicates, g_i)
            return(synth_X[1:-1], synth_y[1:-1])
        else: 
            print("class is out of treshhold boundary. passing")
            pass

    def fit_resample(self, X, y):
        new_X, new_y = self.adasyn_algorithm(X, y)
        new_X = np.concatenate((new_X, X), axis=0)
        new_y = np.concatenate((new_y, y), axis=0)
        return new_X, new_y