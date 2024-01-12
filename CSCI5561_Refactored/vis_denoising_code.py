import os, cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import normalized_root_mse as rmse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.restoration import denoise_tv_chambolle as tv_denoise
from skimage.restoration import denoise_wavelet as wl_denoise
from skimage.restoration import denoise_nl_means as nlm_denoise
from skimage.restoration import denoise_bilateral as bl_denoise
from skimage.restoration import estimate_sigma
import scipy.ndimage.filters as flt
from anisotropic_diffusion_denoising import denoise_img
from tensorflow.keras.models import load_model
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"

def anisodiff(img, niter=1, kappa=50, gamma=0.1, sigma=0):
    
    # Anisotropic diffusion denoising
    # Based on the python implementation by Alistair Muldal <alistair.muldal@pharm.ox.ac.uk>
    # Kappa controls conduction gradient, Gamma controls diffusion speed

    # initialize output array
    img = img.astype('float32')
    imgout = img.copy()

    # initialize deltas
    deltaS = np.zeros(imgout.shape)
    deltaE = np.zeros(imgout.shape)
    NS = np.zeros(imgout.shape)
    EW = np.zeros(imgout.shape)

    # initialize gradients
    gS = np.ones(imgout.shape)
    gE = np.ones(imgout.shape)

    for ii in np.arange(1,niter):

        # calculate the diffs
        deltaS[:-1,: ] = np.diff(imgout,axis=0)
        deltaE[: ,:-1] = np.diff(imgout,axis=1)
        deltaSf=deltaS
        deltaEf=deltaE
        
        # update matrices with the conduction gradient and the delta
        E = np.exp(-(deltaEf/kappa)**2.)*deltaE
        S = np.exp(-(deltaSf/kappa)**2.)*deltaS
        
        # subtract a copy that has been shifted 'North/West' by
        # 1 pixel (this needs to be here???)
        # according to Alistair, "Just don't ask"
        NS[:] = S
        EW[:] = E
        NS[1:,:] -= S[:-1,:]
        EW[:,1:] -= E[:,:-1]
        # update the image
        imgout += gamma*(NS+EW)

    return imgout

# Regularization parameter
alpha = 0.2

# Gradient and divergence with periodic boundaries
def gradient(x):
    g = np.zeros((x.shape[0],x.shape[1],2))
    g[:,:,0] = np.roll(x,-1,axis=0) - x
    g[:,:,1] = np.roll(x,-1,axis=1) - x
    return g

def divergence(p):
    px = p[:,:,0]
    py = p[:,:,1]
    resx = px - np.roll(px,1,axis=0)
    resy = py - np.roll(py,1,axis=1)
    return -(resx + resy)

def min_func(G, F, K, x):
    return G(x) + F(K(x))

def TV_chambolle_pock(noisy_img):

    # Minimization of F(K*x) + G(x)
    K = gradient
    K.T = divergence
    amp = lambda u : np.sqrt(np.sum(u ** 2,axis=2))
    F = lambda u : alpha * np.sum(amp(u))
    G = lambda x : 1/2 * np.linalg.norm(y-x,'fro') ** 2

    # Proximity operators
    normalize = lambda u : u/np.tile(
        (np.maximum(amp(u), 1e-10))[:,:,np.newaxis],
        (1,1,2))
    proxF = lambda u,tau : np.tile(
        soft_thresholding(amp(u), alpha*tau)[:,:,np.newaxis],
        (1,1,2) )* normalize(u)
    proxFS = dual_prox(proxF)
    proxG = lambda x,tau : (x + tau*noisy_img) / (1+tau)

    callback = lambda x : min_func(G, F, K, x)

    img_out, cx = pp.admm(proxFS, proxG, K, noisy_img,
             maxiter=20, full_output=1, callback=callback)


DATASET_PATH = 'BRATS_2018_ALL'
TRAIN_X_PATH = os.path.join(DATASET_PATH, 'X_train_input')
VALID_X_PATH = os.path.join(DATASET_PATH, 'X_test_input')
TRAIN_Y_PATH = os.path.join(DATASET_PATH, 'X_train_target')
VALID_Y_PATH = os.path.join(DATASET_PATH, 'X_test_target')

def image_loader(mode, x_or_y, index):
    """
    Return the numpy image given the information.
    mode: 'train', 'valid' or 'test'
    x_or_y: 'x' or 'y', 'x' stands for the input and 'y' stands for the target
    index: int, the index of the image
    """
    if mode == 'train':
        if x_or_y == 'x':
            filepath = os.path.join(TRAIN_X_PATH, os.path.basename(TRAIN_X_PATH)+'_'+str(index)+'.npy')
        if x_or_y == 'y':
            filepath = os.path.join(TRAIN_Y_PATH, os.path.basename(TRAIN_Y_PATH)+'_'+str(index)+'.npy')
    elif mode == 'valid':
        if x_or_y == 'x':
            filepath = os.path.join(VALID_X_PATH, os.path.basename(VALID_X_PATH)+'_'+str(index)+'.npy')
        if x_or_y == 'y':
            filepath = os.path.join(VALID_Y_PATH, os.path.basename(VALID_Y_PATH)+'_'+str(index)+'.npy')
    else:
        raise ValueError("The first or the second parameter is not valid")
        
    if not isinstance(index, int):
        raise TypeError("Index should be an integer")
    
    if mode == 'train' and (index < 0 or index > 35339):
            raise IndexError("Image index out of range 0 - 35339")
            
    if mode == 'valid' and (index < 0 or index > 8834):
            raise IndexError("Image index out of range 0 - 8834")
    
    return np.load(filepath)

def gen_noisy_image(img, mode):
    mean = 0
    sigma = random.uniform(0.2, 1)
    scale = random.uniform(0.05, 0.2)
    dist = np.random.normal(mean, sigma, img.shape)
    if random.uniform(0, 1) > 0.5:
      noisy_img = img + scale * dist
    else:
      noisy_img = img + scale * (img * dist)
    return noisy_img

# get some data in main memory for visualizing
def get_data(n_samples_train = 500, n_samples_test = 500, overwrite = False):
  if not overwrite:
    x_train = np.load("x_train.npy")
    x_train_target = np.load("x_train_targ.npy")
    x_test = np.load("x_test.npy")
    x_test_target = np.load("x_test_targ.npy")
  else:
    x_train = np.zeros((n_samples_train, 240, 240, 4))
    x_train_target = np.zeros((n_samples_train, 240, 240, 4))
    x_test = np.zeros((n_samples_test, 240, 240, 4))
    x_test_target = np.zeros((n_samples_test, 240, 240, 4))
    for x in tqdm(range(x_train.shape[0]), 'Train data'):
      x_train[x] = image_loader('train', 'x', x)
      x_train_target[x] = gen_noisy_image(image_loader('train', 'x', x), 'train')
    for x in tqdm(range(x_test.shape[0]), 'Test data'):
      x_test_target[x] = image_loader('valid', 'x', x)
      x_test[x] = gen_noisy_image(image_loader('valid', 'x', x), 'valid')
    np.save("./x_train.npy", x_train)
    np.save("./x_train_targ.npy", x_train_target)
    np.save("./x_test.npy", x_test)
    np.save("./x_test_targ.npy", x_test_target)
  return x_train, x_train_target, x_test, x_test_target

x_train, x_train_target, x_test, x_test_target = get_data()

def filter_image(im, filter):
    im = np.pad(im, 1, mode='reflect') # reflect better preserves derivatives
    s = filter.shape + tuple(np.subtract(im.shape, filter.shape) + 1)
    strd = np.lib.stride_tricks.as_strided
    subM = strd(im, shape = s, strides = im.strides * 2)
    return np.einsum('ij,ijkl->kl', filter, subM)

def denoise_img2(img):
    img_out = np.zeros(img.shape)
    for x in range(img.shape[2]):
        img_out[:,:,x] = anisodiff(img[:,:,x],niter=50,kappa=80,gamma=0.035)
        # sharpen output
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        img_out[:,:,x] = filter_image(img_out[:,:,x], kernel)
    
    return img_out

n = 4
offset = 80
f, ax = plt.subplots(n, 4)

stats = np.zeros((2, 2, x_test.shape[0]))
denoising_model = load_model('denoising-model', compile=False)
CBDn = denoising_model.predict(x_test)
for i, x in tqdm(enumerate(x_test), total=500):
    CBD = CBDn[i]
    AD = denoise_img(x)
    original = x_test_target[i]
    stats[0, :, i] = psnr(original, AD, data_range = AD.max() - AD.min()), ssim(original, AD, data_range = AD.max() - AD.min(), channel_axis = 2)
    stats[1, :, i] = psnr(original, CBD, data_range = CBD.max() - CBD.min()), ssim(original, CBD, data_range = CBD.max() - CBD.min(), channel_axis = 2)

# top row is AD, bottom row is NN
print(np.mean(stats, axis=2))




#CBDn = denoising_model.predict(np.array([x_test[0]]))
for i in range(n):
    original = x_test_target[i + offset]
    noisy = gen_noisy_image(x_test_target[i + offset], 'v')
    sigma_est = np.mean(estimate_sigma(noisy, channel_axis=-1))
    TV = tv_denoise(noisy, weight=0.1, channel_axis=2)
    WL = wl_denoise(noisy, channel_axis=2)
    NLM = nlm_denoise(noisy, h=sigma_est*1.5, sigma=sigma_est, channel_axis=2)
    BL = bl_denoise(noisy, channel_axis=2)
    AD = denoise_img(noisy)
    CBD = denoising_model.predict(noisy.reshape((1, *noisy.shape)))[0]
    
    ax[0, i].imshow(noisy)
    ax[0, i].set_title('Noisy')
    ax[0, i].set_xlabel(f'PSNR: {psnr(original, noisy, data_range = noisy.max() - noisy.min()):.3f}, SSIM: {ssim(original, noisy, data_range = noisy.max() - noisy.min(), channel_axis = 2):.3f}')

    #ax[i, 1].imshow(NLM)
    #ax[i, 1].set_title('Non Local Means (skimage)')
    #ax[i, 1].set_xlabel(f'PSNR: {psnr(original, NLM, data_range = NLM.max() - NLM.min()):.3f}, SSIM: {ssim(original, NLM, data_range = NLM.max() - NLM.min(), channel_axis = 2):.3f}')
    
    #ax[i, 2].imshow(TV)
    #ax[i, 2].set_title('Total Variation (skimage)')
    #ax[i, 2].set_xlabel(f' PSNR: {psnr(original, TV, data_range = TV.max() - TV.min()):.3f}, SSIM: {ssim(original, TV, data_range = TV.max() - TV.min(), channel_axis = 2):.3f}')

    ax[1, i].imshow(AD)
    ax[1, i].set_title('AD')
    ax[1, i].set_xlabel(f'PSNR: {psnr(original, AD, data_range = AD.max() - AD.min()):.3f}, SSIM: {ssim(original, AD, data_range = AD.max() - AD.min(), channel_axis = 2):.3f}')

    ax[2, i].imshow(CBD)
    ax[2, i].set_title('NN')
    ax[2, i].set_xlabel(f'PSNR: {psnr(original, CBD, data_range = CBD.max() - CBD.min()):.3f}, SSIM: {ssim(original, CBD, data_range = CBD.max() - CBD.min(), channel_axis = 2):.3f}')

    ax[3, i].imshow(x_test_target[i + offset])
    ax[3, i].set_title('Original')

#[ax.set_axis_off() for ax in ax.ravel()]
plt.tight_layout()
plt.show()

