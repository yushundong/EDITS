

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from scipy.stats import wasserstein_distance
import numpy as np
import math
from scipy import sparse


def metric_wd(feature, adj_norm, flag, weakening_factor, max_hop):

    feature = (feature / feature.norm(dim=0)).detach().cpu().numpy()
    adj_norm = (0.5 * adj_norm + 0.5 * sparse.eye(adj_norm.shape[0])).toarray()  # lambda_{max} = 2
    emd_distances = []
    cumulation = np.zeros_like(feature)

    if max_hop == 0:
        cumulation = feature
    else:
        for i in range(max_hop):
            cumulation += pow(weakening_factor, i) * adj_norm.dot(feature)

    for i in range(feature.shape[1]):
        class_1 = cumulation[torch.eq(flag, 0), i]
        class_2 = cumulation[torch.eq(flag, 1), i]
        emd = wasserstein_distance(class_1, class_2)
        emd_distances.append(emd)

    emd_distances = [0 if math.isnan(x) else x for x in emd_distances]

    if max_hop == 0:
        print('Attribute bias : ')
    else:
        print('Structural bias : ')

    print("Sum of all Wasserstein distance value across feature dimensions: " + str(sum(emd_distances)))
    print("Average of all Wasserstein distance value across feature dimensions: " + str(np.mean(np.array(emd_distances))))

    sns.distplot(np.array(emd_distances).squeeze(), rug=True, hist=True, label='EMD value distribution')
    plt.legend()
    # plt.show()

    num_list1 = emd_distances
    x = range(len(num_list1))

    plt.bar(x, height=num_list1, width=0.4, alpha=0.8, label="Wasserstein distance on reachability")
    plt.ylabel("Wasserstein distance")
    plt.legend()
    # plt.show()


    return emd_distances






