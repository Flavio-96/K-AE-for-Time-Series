from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score \
    , adjusted_mutual_info_score, v_measure_score, fowlkes_mallows_score
import numpy as np


# Dunn Index is heavy and good for small datasets. TODO

# Misura la densità del clustering, ovvero quanto un sample è simile agli altri punto dello stesso cluster
# e quanto bene dista dal cluster più vicino usando una metrica di similarità (euclidea, cosine, ecc).
# Questo score è la media di tutti i silhouette score di ciascun sample
# DTW is fine for TS but it takes too long
def silhouette(dataset, clustering):
    return silhouette_score(dataset, clustering, metric='euclidean')


# DB: Misura la separazione tra cluster, compiendo una media artimetica delle similarità tra coppie di cluster più simili
# , usandodi una misura di similarità tra cluster ad hoc che mette a rapporto la somma dei diametri
# dei cluster (media distanza euclidea intra-cluster) e la distanza euclidea tra i rispettivi centroidi.
# Più tende a 0 meglio è. Fvorisce cluster densi e ben distanti
# Più veloce di silhouette ma limitato alla distanza euclidea
def db(dataset, clustering):
    return davies_bouldin_score(dataset, clustering)


# Righe le label e colonne i cluster
def cm(y_true, y_pred):
    return contingency_matrix(y_true, y_pred)


# Media tra tutti i cluster del numero di sample della label più presente di ciascun cluster.
# Da' una misura di quanto bene il clustering copre il labelling. Se è 1, il clustering ha coperto tutte le label
# , anche ricorrendo ad un numero di cluster maggiore delle classi
def purity(y_true, y_pred):
    cont_matrix = cm(y_true, y_pred)
    return np.sum(np.amax(cont_matrix, axis=0)) / np.sum(cont_matrix)


def rel_purity(y_true, y_pred):
    cm = contingency_matrix(y_true, y_pred)
    labels_sum = np.sum(cm, axis=1)
    rm = np.zeros(cm.shape)
    for j in range(cm.shape[1]):
        for i in range(cm.shape[0]):
            rm[i][j] = cm[i][j] / labels_sum[i]
    # print("Relative Contingency Matrix")
    # print(rm)
    # print(np.max(rm, axis=0))

    max_indexes = np.argmax(rm, axis=0)
    # print(max_indexes)
    sum = 0
    for j in range(rm.shape[1]):
        sum += cm[max_indexes[j]][j]
    return sum / np.sum(cm)


# ARI: fix dell'RI, che mette a rapporto il numero di true (se due sample sono nello stesso cluster allora hanno la stessa label
# + se due sample sono in cluster diversi allora hanno diversa label) sul numero totaale di coppie non ordinate di sample
# va bene quando si vuole un clustering molto fedele al labelling del dataset. Valida per dataset i cui sample appartengono a classi ben distanti.
# Immune al random labelling: https://scikit-learn.org/stable/auto_examples/cluster/plot_adjusted_for_chance_measures.html#sphx-glr-auto-examples-cluster-plot-adjusted-for-chance-measures-py
# Rule of thumb: Use ARI when the ground truth clustering has large equal sized clusters
def ari(y_true, y_pred):
    return adjusted_rand_score(y_true, y_pred)


# FMS: Media geometrica di precision e recall pairwise
def fmi(y_true, y_pred):
    return fowlkes_mallows_score(y_true, y_pred)


# AMIS: fix del MIS, basata sull'entropia di Von Neuman, calcolata per le label e per i cluster
# Immune al random labelling
# Rule of thumb: Usa AMI when the ground truth clustering is unbalanced and there exist small clusters
def ami(y_true, y_pred):
    return adjusted_mutual_info_score(y_true, y_pred, average_method='arithmetic')


# Media armonica di Homogeneity e Completeness.
# Homogeneity: Quanto un cluster ha sample di una sola label
# Completeness: Quanto i sample di una label stanno in un solo cluster
# Entrambi basati sull'entropia di Von Neumann
# Debole al random clustering con alto numero di cluster. Non buono con dataset piccoli e/o grande numero di cluster
def vm(y_true, y_pred):
    return v_measure_score(y_true, y_pred)
