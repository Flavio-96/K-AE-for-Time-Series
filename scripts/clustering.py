from sklearn.cluster import KMeans
from itertools import groupby

def getClustering(features, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    res = kmeans.fit_predict(features)
    kmeans.labels_ +=1
    res = res -1
    groups = groupby(sorted(zip(tuple(res), tuple(labelsTest))), lambda x: x[0])
    counts = {k: dict(Counter(x[1] for x in g)) for k, g in groups}
    return res, counts