from sklearn.cluster import KMeans


def getClustering(features, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=100)
    clustering = kmeans.fit_predict(features)
    kmeans.labels_ += 1
    clustering = clustering -1
    return clustering