from multiprocessing.pool import Pool


from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

from typing import Literal
import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.manifold import MDS

import seaborn as sns
import matplotlib.pyplot as plt
from  mpl_toolkits.mplot3d import Axes3D
from adjustText import adjust_text

from tqdm import tqdm

def _get_JSD_distance(i, j, source_df):
    P_i, P_j = source_df.iloc[i].to_numpy(), source_df.iloc[j].to_numpy()
    M = (P_i + P_j)/2
    return i, j , np.sqrt((entropy(P_i,M, base=2)+entropy(P_j,M, base=2))/2)
    # dist_matrix[i][j] = np.sqrt((entropy(P_i,M, base=2)+entropy(P_j,M, base=2))/2)
    # dist_matrix[j][i] = dist_matrix[i][j]

def multiprocessing_get_dist_matrix(source_df):
    
    dist_matrix_size = source_df.shape[0]
    dist_matrix = np.zeros((dist_matrix_size,dist_matrix_size), dtype=source_df.dtypes.mode()[0].name.lower())
    
    with ProcessPoolExecutor() as executor:
        inputs = [(i, j) for i in range(dist_matrix_size) for j in range(i+1,dist_matrix_size)]
        z_inputs = list(zip(*inputs))
        for i, j, dist in tqdm(executor.map(_get_JSD_distance, *z_inputs, repeat(source_df))):
            dist_matrix[i][j] = dist
            dist_matrix[j][i] = dist_matrix[i][j]
    
    # with Pool() as pool:
    #     inputs = [(i, j, source_df) for i in range(dist_matrix_size) for j in range(i+1,dist_matrix_size)]
    #     for i, j, dist in pool.starmap(_get_JSD_distance, inputs):
    #         dist_matrix[i][j] = dist
    #         dist_matrix[j][i] = dist_matrix[i][j]

    return dist_matrix


def get_dist_matrix(source_df:pd.DataFrame) -> np.ndarray:
    
    dist_matrix_size = source_df.shape[0]
    dist_matrix = np.zeros((dist_matrix_size,dist_matrix_size), dtype=source_df.dtypes.mode()[0].name.lower())
    
    for i in tqdm(range(dist_matrix_size)):
        for j in range(i+1,dist_matrix_size):
            P_i, P_j = source_df.iloc[i].to_numpy(), source_df.iloc[j].to_numpy()
            M = (P_i + P_j)/2
            # print('i:', i, '  j:', j)
            dist_matrix[i][j] = np.sqrt((entropy(P_i,M, base=2)+entropy(P_j,M, base=2))/2)
            dist_matrix[j][i] = dist_matrix[i][j]

    return dist_matrix

def get_IGT_embeddings(dist_matrix:np.ndarray) -> np.ndarray:
    embedding = MDS(n_components=dist_matrix.shape[0]-1, dissimilarity='precomputed')
    return embedding.fit_transform(X=dist_matrix)

def get_3D_IGT_plot(dist_embeddings:np.ndarray, point_labels:None|np.ndarray = None) -> plt.Figure:
    fig = plt.figure(figsize=(16,10))
    ax= Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)

    ax.scatter3D(
        dist_embeddings[:,0], 
        dist_embeddings[:,1], 
        dist_embeddings[:,2], # type: ignore
                )
    if point_labels is not None:
        [ax.text(i, j, k, s) for i, j, k, s in zip(
            dist_embeddings[:,0],
            dist_embeddings[:,1], 
            dist_embeddings[:,2],
            point_labels)
        ]
        # texts = [ax.text(i, j, k, s) for i, j, k, s in zip(
        #     dist_embeddings[:,0],
        #     dist_embeddings[:,1], 
        #     dist_embeddings[:,2],
        #     point_labels)
        # ]
        
        # adjust_text(texts, only_move={'points':'y', 'texts':'y'}, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))
    
    ax.plot3D(dist_embeddings[:,0], 
            dist_embeddings[:,1], 
            dist_embeddings[:,2])
    

    return fig

def get_2D_IGT_plot(dist_embeddings:np.ndarray, point_labels:None|np.ndarray = None) -> plt.Figure:
    fig = plt.figure(figsize=(16,10))
    ax = fig.add_subplot()
    ax.scatter(dist_embeddings[:,0], dist_embeddings[:,1])
    
    if point_labels is not None:
        texts = [plt.text(i, j, s) for i, j, s in zip(
            dist_embeddings[:,0],
            dist_embeddings[:,1], 
            point_labels)
        ]
            
        adjust_text(texts, only_move={'points':'y', 'texts':'xy'}, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))
    ax.plot(dist_embeddings[:,0], dist_embeddings[:,1])
    return fig

def igt_plot(source_df:pd.DataFrame, type:Literal['3D', '2D']) -> plt.Figure:
    dist_matrix = get_dist_matrix(source_df)
    dist_embeddings = get_IGT_embeddings(dist_matrix)
    if type == '3D':
        return get_3D_IGT_plot(dist_embeddings)
    else:
        return get_2D_IGT_plot(dist_embeddings)
    
