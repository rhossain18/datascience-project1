import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score

n_clusters =10
#f = open('mnist_train.csv','r')
train_data = pd.read_csv(r'mnist_train.csv')
Y_train = np.array(train_data['label'])
X_train = np.array(train_data.drop('label', axis=1))



#Principal Component Analysis Of The Data For Higher Performance Metrics
Xn = normalize(X_train,axis=0)
pca = PCA(n_components=2)
X = pca.fit_transform(Xn)


no_of_subplots = 4
plt.figure(figsize=(10, no_of_subplots))
plot_no=0
for linkage in ['complete','average','ward','actual_data']:
    if linkage=='actual_data':
        pass
    else:
        model = AgglomerativeClustering(linkage=linkage, n_clusters=n_clusters)
        model.fit(X)
        labels = model.labels_
    
    plot_no+=1
    plt.subplot(1, no_of_subplots, plot_no)
    plt.scatter(X[:, 0], X[:, 1], c=labels)
    plt.title('linkage=%s' % (linkage))
    print('linkage:',linkage)
    print(labels,"\n")
    
#Calculating The Performance Metrics Of Agglomerative Clustering 
    #printing silhouette score using manhattan metric
    print('Manhattan/L1 Distance')
    print(silhouette_score(X,labels, metric='manhattan'))
    #printing silhouette score using L2 metric
    print('L2 Distance')
    print(silhouette_score(X,labels, metric='l2'))
    #printing silhouette score using Cosine metric
    print('Cosine Distance')
    print(silhouette_score(X,labels, metric='cosine'))


plt.show()