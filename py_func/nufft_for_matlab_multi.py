import numpy as np
import torch
import torchkbnufft as tkbn
from scipy import ndimage
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


batch_size = 500
im_size = None
L = None
grid_size = None
ktraj = None
nufft_ob = None
norm = 'ortho'
batch_index = None
adjnufft_ob = None
toep_op = None
numpoints = 8  # 越小越快
dcomp = None
ncoil = 32
kernel = None


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


def mrisensesim(size, ncoils=8, array_cent=None, coil_width=2, n_rings=None, phi=0):
    """Apply simulated sensitivity maps. Based on a script by Florian Knoll.

    Args:
        size (tuple): Size of the image array for the sensitivity coils.
        nc_range (int, default: 8): Number of coils to simulate.
        array_cent (tuple, default: 0): Location of the center of the coil
            array.
        coil_width (double, default: 2): Parameter governing the width of the
            coil, multiplied by actual image dimension.
        n_rings (int, default: ncoils // 4): Number of rings for a
            cylindrical hardware set-up.
        phi (double, default: 0): Parameter for rotating coil geometry.

    Returns:
        list: A list of dimensions (ncoils, (N)), specifying spatially-varying
            sensitivity maps for each coil.
    """
    if array_cent is None:
        c_shift = [0, 0, 0]
    elif len(array_cent) < 3:
        c_shift = array_cent + (0,)
    else:
        c_shift = array_cent

    c_width = coil_width * min(size)

    if len(size) > 2:
        if n_rings is None:
            n_rings = ncoils // 4

    c_rad = min(size[0:1]) / 2
    smap = []
    if len(size) > 2:
        zz, yy, xx = np.meshgrid(
            range(size[2]), range(size[1]), range(size[0]), indexing="ij"
        )
    else:
        yy, xx = np.meshgrid(range(size[1]), range(size[0]), indexing="ij")

    if ncoils > 1:
        x0 = np.zeros((ncoils,))
        y0 = np.zeros((ncoils,))
        z0 = np.zeros((ncoils,))

        for i in range(ncoils):
            if len(size) > 2:
                theta = np.radians((i - 1) * 360 / (ncoils + n_rings) + phi)
            else:
                theta = np.radians((i - 1) * 360 / ncoils + phi)
            x0[i] = c_rad * np.cos(theta) + size[0] / 2
            y0[i] = c_rad * np.sin(theta) + size[1] / 2
            if len(size) > 2:
                z0[i] = (size[2] / (n_rings + 1)) * (i // n_rings)
                smap.append(
                    np.exp(
                        -1
                        * ((xx - x0[i]) ** 2 + (yy - y0[i]) ** 2 + (zz - z0[i]) ** 2)
                        / (2 * c_width)
                    )
                )
            else:
                smap.append(
                    np.exp(-1 * ((xx - x0[i]) ** 2 +
                           (yy - y0[i]) ** 2) / (2 * c_width))
                )
    else:
        x0 = c_shift[0]
        y0 = c_shift[1]
        z0 = c_shift[2]
        if len(size) > 2:
            smap = np.exp(
                -1 * ((xx - x0) ** 2 + (yy - y0) ** 2 +
                      (zz - z0) ** 2) / (2 * c_width)
            )
        else:
            smap = np.exp(-1 * ((xx - x0) ** 2 +
                          (yy - y0) ** 2) / (2 * c_width))

    side_mat = np.arange(int(size[0] // 2) - 20, 1, -1)
    side_mat = np.reshape(side_mat, (1,) + side_mat.shape) * \
        np.ones(shape=(size[1], 1))
    cent_zeros = np.zeros(shape=(size[1], size[0] - side_mat.shape[1] * 2))

    ph = np.concatenate((side_mat, cent_zeros, side_mat), axis=1) / 10
    if len(size) > 2:
        ph = np.reshape(ph, (1,) + ph.shape)

    for i, s in enumerate(smap):
        smap[i] = s * np.exp(i * 1j * ph * np.pi / 180)

    return smap


def calculate_csm_inati_iter(im, smoothing=5, niter=5, thresh=1e-3,
                             verbose=False):
    """ Fast, iterative coil map estimation for 2D or 3D acquisitions.

    Parameters
    ----------
    im : ndarray
        Input images, [coil, y, x] or [coil, z, y, x].
    smoothing : int or ndarray-like
        Smoothing block size(s) for the spatial axes.
    niter : int
        Maximal number of iterations to run.
    thresh : float
        Threshold on the relative coil map change required for early
        termination of iterations.  If ``thresh=0``, the threshold check
        will be skipped and all ``niter`` iterations will be performed.
    verbose : bool
        If true, progress information will be printed out at each iteration.

    Returns
    -------
    coil_map : ndarray
        Relative coil sensitivity maps, [coil, y, x] or [coil, z, y, x].
    coil_combined : ndarray
        The coil combined image volume, [y, x] or [z, y, x].

    Notes
    -----
    The implementation corresponds to the algorithm described in [1]_ and is a
    port of Gadgetron's ``coil_map_3d_Inati_Iter`` routine.

    For non-isotropic voxels it may be desirable to use non-uniform smoothing
    kernel sizes, so a length 3 array of smoothings is also supported.

    References
    ----------
    .. [1] S Inati, MS Hansen, P Kellman.  A Fast Optimal Method for Coil
        Sensitivity Estimation and Adaptive Coil Combination for Complex
        Images.  In: ISMRM proceedings; Milan, Italy; 2014; p. 4407.
    """

    im = np.asarray(im)
    if im.ndim < 3 or im.ndim > 4:
        raise ValueError("Expected 3D [ncoils, ny, nx] or 4D "
                         " [ncoils, nz, ny, nx] input.")

    if im.ndim == 3:
        # pad to size 1 on z for 2D + coils case
        images_are_2D = True
        im = im[:, np.newaxis, :, :]
    else:
        images_are_2D = False

    # convert smoothing kernel to array
    if isinstance(smoothing, int):
        smoothing = np.asarray([smoothing, ] * 3)
    smoothing = np.asarray(smoothing)
    if smoothing.ndim > 1 or smoothing.size != 3:
        raise ValueError("smoothing should be an int or a 3-element 1D array")

    if images_are_2D:
        smoothing[2] = 1  # no smoothing along z in 2D case

    # smoothing kernel is size 1 on the coil axis
    smoothing = np.concatenate(([1, ], smoothing), axis=0)

    ncha = im.shape[0]

    try:
        # numpy >= 1.7 required for this notation
        D_sum = im.sum(axis=(1, 2, 3))
    except:
        D_sum = im.reshape(ncha, -1).sum(axis=1)

    v = 1/np.linalg.norm(D_sum)
    D_sum *= v
    R = 0

    for cha in range(ncha):
        R += np.conj(D_sum[cha]) * im[cha, ...]

    eps = np.finfo(im.real.dtype).eps * np.abs(im).mean()
    for it in range(niter):
        if verbose:
            print("Coil map estimation: iteration %d of %d" % (it+1, niter))
        if thresh > 0:
            prevR = R.copy()
        R = np.conj(R)
        coil_map = im * R[np.newaxis, ...]
        coil_map_conv = smooth(coil_map, box=smoothing)
        D = coil_map_conv * np.conj(coil_map_conv)
        R = D.sum(axis=0)
        R = np.sqrt(R) + eps
        R = 1/R
        coil_map = coil_map_conv * R[np.newaxis, ...]
        D = im * np.conj(coil_map)
        R = D.sum(axis=0)
        D = coil_map * R[np.newaxis, ...]
        try:
            # numpy >= 1.7 required for this notation
            D_sum = D.sum(axis=(1, 2, 3))
        except:
            D_sum = im.reshape(ncha, -1).sum(axis=1)
        v = 1/np.linalg.norm(D_sum)
        D_sum *= v

        imT = 0
        for cha in range(ncha):
            imT += np.conj(D_sum[cha]) * coil_map[cha, ...]
        magT = np.abs(imT) + eps
        imT /= magT
        R = R * imT
        imT = np.conj(imT)
        coil_map = coil_map * imT[np.newaxis, ...]

        if thresh > 0:
            diffR = R - prevR
            vRatio = np.linalg.norm(diffR) / np.linalg.norm(R)
            if verbose:
                print("vRatio = {}".format(vRatio))
            if vRatio < thresh:
                break

    if images_are_2D:
        # remove singleton z dimension that was added for the 2D case
        coil_map = coil_map[:, 0, :, :]

    return coil_map


def smooth(img, box=5):
    '''Smooths coil images

    :param img: Input complex images, ``[y, x] or [z, y, x]``
    :param box: Smoothing block size (default ``5``)

    :returns simg: Smoothed complex image ``[y,x] or [z,y,x]``
    '''
    t_real = np.zeros(img.shape)
    t_imag = np.zeros(img.shape)

    ndimage.filters.uniform_filter(img.real, size=box, output=t_real)
    ndimage.filters.uniform_filter(img.imag, size=box, output=t_imag)
    simg = t_real + 1j*t_imag
    return simg


def csm_init():
    '''
    coil sensitivity maps initialization
    '''
    smap_init = np.absolute(
        np.stack(mrisensesim(im_size, ncoils=ncoil, coil_width=32)))
    smap_init = torch.tensor(smap_init).unsqueeze(0).type(dtype_complex)
    return {'smap_init': smap_init.cpu().numpy()}


def csm_estimat(coil_images, smoothing=5, niter=5, thresh=1e-3, verbose=False):
    '''
    coil sensitivity maps estimation
    '''
    csm_est = calculate_csm_inati_iter(
        coil_images, smoothing=smoothing, niter=niter, thresh=thresh, verbose=verbose)
    return {'csm_est': csm_est}
 

def init_nufft_multi_op(batch_size_init=50, Nx_init=None, Ny_init=None, L_init=None, grid_factor_init=None, ktraj_init=None):
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
        # interp_mats = tkbn.calc_tensor_spmatrix(ktraj,im_size,grid_size)
        # interp_mats = tuple([t.type(dtype_float).to(device) for t in interp_mats])
        nufft_ob = tkbn.KbNufft(
            im_size=im_size, grid_size=grid_size, numpoints=numpoints).to(device)
        adjnufft_ob = tkbn.KbNufftAdjoint(
            im_size=im_size, grid_size=grid_size, numpoints=numpoints).to(device)
        if batch_size >= L:
            batch_index = [[0, L]]
        else:
            batch_index = get_batch_index(L, batch_size)
        dcomp = tkbn.calc_density_compensation_function(
            ktraj=ktraj, im_size=im_size)
        kernel = tkbn.calc_toeplitz_kernel(
                        ktraj, im_size, weights=dcomp, norm=norm, grid_size=grid_size, numpoints=numpoints).to(device)
        toep_op = tkbn.ToepNufft().to(device)
        print('nufft operator initialized')


def nufft_forward_multi_op(x=None, smap=None):
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
        if smap is not None:
            smap = torch.tensor(smap).type(dtype_complex).to(device)
        k_samples = np.zeros((L, ncoil, ktraj.shape[2]), dtype=np.complex64)
        for i in batch_index:
            im_k_samples = nufft_ob(x[i[0]:i[1], :, :, :],
                                    ktraj[i[0]:i[1], :, :], smaps=smap, norm=norm)
            k_samples[i[0]:i[1], :, :] = im_k_samples.cpu().numpy()
        # k_samples = k_samples/np.max(np.abs(k_samples))
    return {'k_samples': k_samples}


def nufft_adjoint_multi_op(k_samples=None, smap=None):
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
        if smap is not None:
            smap = torch.tensor(smap).type(dtype_complex).to(device)
        for i in batch_index:
            im_x = adjnufft_ob(k_samples[i[0]:i[1], :, :]*dcomp[i[0]:i[1], :, :],
                            ktraj[i[0]:i[1], :, :], smaps=smap, norm=norm)
            x[:, :, i[0]:i[1]] = np.squeeze(im_x.permute(1, 2, 3, 0).cpu().numpy())
        # if np.max(np.abs(x))>0:
        #     x = x/np.max(np.abs(x))
    return {'x': x}


def nufft_ata_multi_op(x=None, smap=None, ):
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
        if smap is not None:
            smap = torch.tensor(smap).type(dtype_complex).to(device)
        for i in batch_index:
            im_k_samples = nufft_ob(x[i[0]:i[1], :, :, :],
                                    ktraj[i[0]:i[1], :, :], smaps=smap, norm=norm)  # sampling
            im_x = adjnufft_ob(
                im_k_samples*dcomp[i[0]:i[1], :, :], ktraj[i[0]:i[1], :, :], smaps=smap, norm=norm)  # adjoint
            x_ata[:, :, i[0]:i[1]] = np.squeeze(
                im_x.permute(1, 2, 3, 0).cpu().numpy())
        # if np.max(np.abs(x_ata)) > 0:
        #     x_ata = x_ata/np.max(np.abs(x_ata))
    return {'x_ata': x_ata}


def nufft_toep_ata_multi_op(x=None, smap=None):
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
        if smap is not None:
            smap = torch.tensor(smap).type(dtype_complex).to(device)
            smap = smap.expand(L, -1, -1, -1)  # [T, ncoil, Nx, Ny]
        for i in batch_index:
            im_x = toep_op(x[i[0]:i[1], :, :, :], kernel[i[0]:i[1], :, :], smaps=smap[i[0]:i[1], :, :, :], norm=norm)  
            x_ata[:, :, i[0]:i[1]] = np.squeeze(
                im_x.permute(1, 2, 3, 0).cpu().numpy())
        # if np.max(np.abs(x_ata)) > 0:
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
    L = 400
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
                       'X_original'], dtype=np.complex64)
    ktraj = np.asarray(h5.File('ktraj_nufft.mat', 'r')[
                       'ktraj_nufft']).astype(complex).transpose(2, 1, 0)
    # init nufft operator
    m_max = np.max(np.abs(image))
    image = image/m_max
    init_nufft_multi_op(batch_size_init=50, Nx_init=im_size[0], Ny_init=im_size[1],
                        L_init=L, grid_factor_init=grid_factor, ktraj_init=ktraj)
    smap_init = csm_init()
    smap_init =smap_init['smap_init']
    # forward backward operator
    start_normal = time.perf_counter()
    k_samples = nufft_forward_multi_op(x=image,smap=smap_init)['k_samples']
    x = nufft_adjoint_multi_op(k_samples=k_samples,smap=smap_init)['x']
    end_normal = time.perf_counter()
    start_ata = time.perf_counter()
    x_ata = nufft_ata_multi_op(x=image,smap=smap_init)['x_ata']
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
