import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

'''
template_mathch: 时域矩阵与字典匹配
build_maps: 由时域矩阵构建参数图
'''


def normlize_data(data):
    '''
    逐行归一化数据
    输入：
    data: 数据矩阵 [L, N]
    输出：
    data: 逐行归一化后的数据矩阵 [L, N]
    norms_D: 每行的范数 [L, 1]
    '''
    norms_D = np.sqrt(np.sum(np.abs(data)**2, axis=1))  # L*1
    norms_D[norms_D == 0] = 1
    norms_D_rep = np.tile(norms_D, (data.shape[1], 1)).T  # L*N
    data = data/norms_D_rep
    return data, norms_D


def template_match(X, D, var=0.9):
    '''
    时域矩阵与字典匹配
    输入：
    X：时域矩阵 L*Pixels
    D：字典 Entries*L
    var: 匹配阈值
    输出：
    match_index：匹配索引矩阵 Pixels*_
    M0：质子密度Pixels*_
    '''
    D_normed, norms_D = normlize_data(D)
    X_normed, _ = normlize_data(X)
    # Calculate inner product
    inner_DX = np.abs(np.inner(D_normed, X_normed))  # Entries*Pixels
    match_var = np.abs(np.max(inner_DX, axis=0))  # _*Pixels 匹配度0-1
    match_index = np.argmax(inner_DX, axis=0)  # _*Pixels 最大匹配位置索引
    norms_D = np.tile(norms_D, (X.shape[0], 1)).T  # Entries*Pixels
    M0 = np.real(np.inner(D_normed, X))
    M0 = M0/norms_D
    M0_maps = np.zeros(M0.shape[1])
    for i in range(M0.shape[1]):
        if match_var[i] < var:
            M0_maps[i] = 0.0
        else:
            M0_maps[i] = M0[match_index[i], i]

    return match_index, M0_maps


def template_match_batch(X, D, LUT, var=0.9):
    '''
    时域矩阵与字典匹配
    输入：
    X：时域矩阵 L*Pixels
    D：字典 Entries*L
    var: 匹配阈值
    输出：
    match_index：匹配索引矩阵 Pixels*_
    M0：质子密度Pixels*_
    '''
    N, L = X.shape
    match_index, M0 = template_match(
        X, D, var=var)
    T1 = np.zeros(M0.shape)
    T2 = np.zeros(M0.shape)
    for i in range(N):
        if M0[i] == 0.0:
            T1[i] = 0.0
            T2[i] = 0.0
        else:
            T1[i] = LUT[match_index[i], 0]
            T2[i] = LUT[match_index[i], 1]
    return T1, T2, M0


def get_batch_index(LEN, batch_size):
    batch_size = int(batch_size)
    if LEN <= batch_size:
        raise ValueError(
            'The LEN should be longer than batch_size.')
    num = LEN//batch_size
    yu = LEN % batch_size
    out = []
    for i in range(num):
        out.append([i*batch_size, (i+1)*batch_size])
    if yu != 0:
        out.append([num*batch_size, LEN])
    return out


def build_maps(X, D, LUT, var, batch_size=10000):
    '''
    由时域矩阵构建参数图
    输入：
    X：时域矩阵 N1*N2*L
    D：字典 Entries*L
    LUT: 构建字典对应的参照表Entries*2
    var: 匹配阈值
    输出：
    T1: T1map N1*N2
    T2: T2map N1*N2
    M0：质子密度map N1*N2
    '''
    N1, N2, L = X.shape
    T1 = np.zeros((N1*N2,))
    T2 = np.zeros((N1*N2,))
    M0 = np.zeros((N1*N2,))
    batch_index = get_batch_index(N1*N2, batch_size)
    X = X.reshape(N1*N2, L)
    for i in batch_index:
        T1[i[0]:i[1], ], T2[i[0]:i[1], ], M0[i[0]:i[1], ] = template_match_batch(
            X[i[0]:i[1], :], D, LUT, var)
    T1 = T1.reshape(N1, N2)
    T2 = T2.reshape(N1, N2)
    M0 = M0.reshape(N1, N2)
    return T1, T2, M0


def build_maps_mat(X, D, LUT, var, batch_size):
    '''
    由时域矩阵构建参数图
    输入：
    X：时域矩阵 N1*N2*L
    D：字典 Entries*L
    LUT: 构建字典对应的参照表Entries*2
    var: 匹配阈值
    输出：
    T1: T1map N1*N2
    T2: T2map N1*N2
    M0：质子密度map N1*N2
    '''
    # X = np.array(X)
    # D = np.array(D)
    # LUT = np.array(LUT)
    T1, T2, M0 = build_maps(X, D, LUT, var, batch_size)
    # T1 = array.array(T1)
    # T2 = array.array(T2)
    # M0 = array.array(M0)
    return {'t1': T1, 't2': T2, 'm0': M0}
