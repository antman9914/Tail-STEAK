import json
import time
import logging
import numpy as np
import linecache
from FastNode2Vec import FastNode2Vec
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
palette = sns.color_palette("bright", 4)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

emb_dim = 32
num_walks = 30
walk_length = 15
window = 3
epochs = 3
p = 1.0
q = 10.0

def dense_feat_extract():
    num_node = 0        # TODO: Pre-compute the amount of node and fill the blank
    base_edge_list = []     # TODO: Specify the adjacency matrix of your graph here
    base_edge_inverted = np.reshape(np.concatenate([base_edge_list[1], base_edge_list[0]], axis=0), (2, -1))
    base_edge = np.concatenate([base_edge_list, base_edge_inverted], axis=-1)
    base_edge = base_edge_list

    n2v_model = FastNode2Vec(num_node, base_edge[0], base_edge[1])
    alpha_schedule = [[1,2,2,301], [0.05, 0.05, 0.005, 0.005]]
    n2v_model.run_node2vec(
        dim=emb_dim, 
        epochs=epochs, 
        num_walks=num_walks, 
        walk_length=walk_length, 
        window=window, 
        alpha_schedule=alpha_schedule,
        p=p, 
        q=q,
    )
    embs = n2v_model.get_embeddings()
    print(embs.shape)
    np.save('init_emb.npy', embs)


dense_feat_extract()