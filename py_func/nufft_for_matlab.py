import numpy as np
import torch
import torchkbnufft as tkbn
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

dtype_float = torch.float32
dtype_complex = torch.complex64
# 根据是否有GPU选择设备
# 若有多个GPU，选择占用显存最少的GPU
if torch.cuda.is_available():
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'], capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')
        free_memory = []
        for line in lines:
            _, free_mem = line.split(',')
            free_memory.append(int(free_mem))
        device_id = np.argmax(free_memory)
        device = torch.device('cuda', device_id)
    except Exception as e:
        print(f"执行 nvidia-smi 命令时出错: {e}，将使用默认GPU设备0")
        device = torch.device('cuda', 0)
else:
    device = torch.device('cpu')


batch_size = 50
im_size = None
L = None
grid_size = None
ktraj = None
nufft_ob = None
norm = 'ortho'
batch_index = None
adjnufft_ob = None
numpoints = 8  # 越小越快
dcomp = None
kernel = None
toep_op = None


def calc_density_compensation_function(ktraj):
    '''
    计算密度补偿函数
    输入：
    ktraj: k采样轨迹 torch Tensor [L, 2, N_samples]
    输出：
    dcomp: 密度补偿函数 torch Tensor [L, 2, N_samples]
    '''
    L, _, N_samples = ktraj.shape
    ktraj = ktraj/2/torch.pi
    dcomp = torch.zeros([L, 1, N_samples]).type(dtype_float).to(device)
    for i in range(L):
        #  l2-normalize k-space trajectory
        dcomp[i, :, :] = torch.sqrt(torch.sum(ktraj[i, :, :]**2, dim=0))
    dcomp = torch.clamp(dcomp, min=0, max=1)
    return dcomp

def get_batch_index(LEN, batch_size):
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


def init_nufft_op(batch_size_init=50, Nx_init=None, Ny_init=None, L_init=None, grid_factor_init=None, ktraj_init=None):
    '''
    NUFFT欠采样算子涉及参数初始化
    输入：
    batch_size: 批处理大小
    im_size: 图像大小 (Nx, Ny)
    L: 图像长度 
    grid_factor: oversampling ratio 默认2
    ktraj: k采样轨迹 numpy数组 [L, 2, N_samples]
    输出：
    None
    '''
    global batch_size, im_size, L, grid_size, ktraj, nufft_ob, batch_index, adjnufft_ob, dcomp, kernel, toep_op
    batch_size = int(batch_size_init)
    if Nx_init is None:
        Nx_init = 220
    if Ny_init is None:
        Ny_init = 220
    im_size = (int(Nx_init), int(Ny_init))
    if L_init is None:
        L_init = 400
    L = int(L_init)
    if grid_factor_init is None:
        grid_factor_init = 2
    grid_size = (int(grid_factor_init*Nx_init), int(grid_factor_init*Ny_init))
    if ktraj_init is None:
        assert False, "ktraj is None"
    with torch.no_grad():
        ktraj = torch.tensor(ktraj_init).type(dtype_float).to(device)
        nufft_ob = tkbn.KbNufft(
            im_size=im_size, grid_size=grid_size, numpoints=numpoints).to(device)
        adjnufft_ob = tkbn.KbNufftAdjoint(
            im_size=im_size, grid_size=grid_size, numpoints=numpoints).to(device)
        if batch_size >= L:
            batch_index = [[0, L]]
        else:
            batch_index = get_batch_index(L, batch_size)
        dcomp = tkbn.calc_density_compensation_function(
            ktraj=ktraj, im_size=im_size, num_iterations=10, grid_size=grid_size, numpoints=numpoints)
        # dcomp = calc_density_compensation_function(ktraj)
        kernel = tkbn.calc_toeplitz_kernel(
                ktraj, im_size, weights=dcomp, norm=norm, grid_size=grid_size, numpoints=numpoints).to(device)
        toep_op = tkbn.ToepNufft().to(device)

        print('nufft operator initialized')


def nufft_forward_op(x=None):
    '''
    NUFFT欠采样
    输入：
    x: 图像numpy矩阵 [Nx, Ny, L]
    输出：
    k_samples: k空间采样数据 [L, 1, N_samples]
    '''
    with torch.no_grad():
        x = torch.tensor(x).type(dtype_complex).to(device)
        x = x.unsqueeze(0).permute(3, 0, 1, 2)
        k_samples = np.zeros((L, 1, ktraj.shape[2]), dtype=np.complex64)
        for i in batch_index:
            im_k_samples = nufft_ob(x[i[0]:i[1], :, :, :],
                                    ktraj[i[0]:i[1], :, :], norm=norm)
            k_samples[i[0]:i[1], :, :] = im_k_samples.cpu().numpy()
    return {'k_samples': k_samples}


def nufft_adjoint_op(k_samples=None):
    '''
    NUFFT欠采样
    输入：
    k_samples: k空间采样数据 [L, 1, N_samples]
    输出：
    x: 图像numpy矩阵 [Nx, Ny, L]
    '''
    with torch.no_grad():
        k_samples = torch.tensor(k_samples).type(dtype_complex).to(device)
        x = np.zeros((im_size[0], im_size[1], L), dtype=np.complex64)
        for i in batch_index:
            im_x = adjnufft_ob(k_samples[i[0]:i[1], :, :]*dcomp[i[0]:i[1], :, :],
                            ktraj[i[0]:i[1], :, :], norm=norm)
            x[:, :, i[0]:i[1]] = np.squeeze(im_x.permute(1, 2, 3, 0).cpu().numpy())
        # if np.max(np.abs(x)) > 0:
        #     x = x/np.max(np.abs(x))
    return {'x': x}


def nufft_ata_op(x=None):
    '''
    NUFFT欠采样
    输入：
    x: 图像numpy矩阵 [Nx, Ny, L]
    输出：
    x_ata: 图像numpy矩阵 [Nx, Ny, L]
    '''
    with torch.no_grad():
        x = torch.tensor(x).type(dtype_complex).to(device)
        x = x.unsqueeze(0).permute(2, 0, 1)
        x_ata = np.zeros((im_size[0], im_size[1], L), dtype=np.complex64)
        for i in batch_index:
            im_k_samples = nufft_ob(x[i[0]:i[1], :, :, :],
                                    ktraj[i[0]:i[1], :, :], norm=norm)  # sampling
            im_x = adjnufft_ob(
                im_k_samples*dcomp[i[0]:i[1], :, :], ktraj[i[0]:i[1], :, :], norm=norm)  # adjoint
            x_ata[:, :, i[0]:i[1]] = np.squeeze(
                im_x.permute(1, 2, 3, 0).cpu().numpy())
        # if np.max(np.abs(x_ata)) > 0:
        #     x_ata = x_ata/np.max(np.abs(x_ata))
    return {'x_ata': x_ata}


def nufft_toep_ata_op(x=None):
    '''
    NUFFT欠采样
    输入：
    x: 图像numpy矩阵 [Nx, Ny, L]
    输出：
    x_ata: 图像numpy矩阵 [Nx, Ny, L]
    '''
    with torch.no_grad():
        x = torch.tensor(x).type(dtype_complex).to(device)
        x = x.unsqueeze(0).permute(3, 0, 1, 2)
        x_ata = np.zeros((im_size[0], im_size[1], L), dtype=np.complex64)
        smap = torch.ones((L, 1, im_size[0], im_size[1])).type(dtype_complex).to(device)
        for i in batch_index:
            im_x = toep_op(x[i[0]:i[1], :, :, :], kernel[i[0]:i[1], :, :], smaps=smap[i[0]:i[1], :, :, :], norm=norm)  
            x_ata[:, :, i[0]:i[1]] = np.squeeze(
                im_x.permute(1, 2, 3, 0).cpu().numpy())
        # if np.max(np.abs(x_ata)) != 0:
        #     x_ata = x_ata/np.max(np.abs(x_ata))
    return {'x_ata': x_ata}


if __name__ == '__main__':
    from scipy.io import loadmat, savemat
    import h5py as h5
    import matplotlib.pyplot as plt
    import time
    import math

    def cal_snr(noise_img, clean_img):
        noise_signal = noise_img-clean_img
        clean_signal = clean_img
        noise_signal_2 = noise_signal**2
        clean_signal_2 = clean_signal**2
        sum1 = np.sum(clean_signal_2)
        sum2 = np.sum(noise_signal_2)
        snrr = 20*math.log10(math.sqrt(sum1)/math.sqrt(sum2))
        return snrr
    grid_factor = 2
    L = 100
    im_size = (220, 220)
    # ktraj = 2*np.asarray(h5.File('ktraj.mat', 'r')['ktraj'])*np.pi
    # image = np.asarray(loadmat('im.mat')['im']).astype(complex)
    # image = torch.tensor(image).unsqueeze(0)
    # image = image.repeat(L, 1, 1)
    # ktraj = torch.tensor(ktraj)
    # ktraj = ktraj.unsqueeze(0).repeat(L, 1, 1)

    # image = image.permute(1, 2, 0).type(dtype_complex).to(device)
    # ktraj = ktraj.type(dtype_float).to(device)

    # ktraj = ktraj.cpu().numpy()
    # image = image.cpu().numpy()

    image = np.asarray(loadmat('X_original.mat')[
                       'X_original'], dtype=np.complex64)[:,:,:L]
    ktraj = np.asarray(h5.File('ktraj_nufft.mat', 'r')[
                       'ktraj_nufft']).astype(complex).transpose(2, 1, 0)[:L,:,:]
    # init nufft operator
    m_max = np.max(np.abs(image))
    image = image/m_max
    init_nufft_op(batch_size_init=10, Nx_init=im_size[0], Ny_init=im_size[1],
                  L_init=L, grid_factor_init=grid_factor, ktraj_init=ktraj)
    # forward backward operator
    start_normal = time.perf_counter()
    k_samples = nufft_forward_op(x=image)['k_samples']
    x = nufft_adjoint_op(k_samples=k_samples)['x']
    end_normal = time.perf_counter()
    start_ata = time.perf_counter()
    x_ata = nufft_toep_ata_op(x=image)['x_ata']
    end_ata = time.perf_counter()
    print('forward/adjoint, normal time: {}, ata time: {}'.format(
        end_normal-start_normal, end_ata-start_ata))
    snr_recon = cal_snr(x, image)
    plt.figure()
    plt.subplot(131)
    plt.imshow(np.abs(image[:, :, 0]))
    plt.gray()
    plt.subplot(132)
    plt.imshow(np.abs(x[:, :, 0]))
    plt.gray()
    plt.subplot(133)
    plt.imshow(np.abs(x_ata[:, :, 0]))
    plt.gray()
    plt.show()
    print(snr_recon)
