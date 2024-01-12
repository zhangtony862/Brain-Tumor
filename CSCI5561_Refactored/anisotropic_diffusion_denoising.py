import numpy as np
import cv2
from skimage.restoration import denoise_tv_chambolle as tv_denoise
from skimage.restoration import denoise_wavelet as wl_denoise

def anisodiff(img, niter=1, kappa=50, gamma=0.1):
    
    # Anisotropic diffusion denoising
    # Based on the python2 implementation by Alistair Muldal <alistair.muldal@pharm.ox.ac.uk>
    # Kappa controls conduction gradient, Gamma controls diffusion speed

    # initialize output array
    img = img.astype('float32')
    imgout = img.copy()

    # initialize deltas
    deltaS = np.zeros(imgout.shape)
    deltaE = np.zeros(imgout.shape)
    NS = np.zeros(imgout.shape)
    EW = np.zeros(imgout.shape)

    for ii in np.arange(1,niter):

        # calculate the diffs
        deltaS[:-1,: ] = np.diff(imgout,axis=0)
        deltaE[: ,:-1] = np.diff(imgout,axis=1)
        deltaSf=deltaS
        deltaEf=deltaE
        
        # update matrices with the conduction gradient and the delta
        E = np.exp(-(deltaEf/kappa)**2.)*deltaE
        S = np.exp(-(deltaSf/kappa)**2.)*deltaS
        
        # Subtract a copy that has been shifted 'North/West' by 1 pixel
        # (This needs to be here??? Breaks without it, not sure why)
        # "Don't [ask] questions. just do it. trust me." - Alistair
        NS[:] = S
        EW[:] = E
        NS[1:,:] -= S[:-1,:]
        EW[:,1:] -= E[:,:-1]
        # update the image
        imgout += gamma*(NS+EW)

    return imgout

def filter_image(im, filter):
    im = np.pad(im, 1, mode='reflect') # reflect better preserves derivatives
    s = filter.shape + tuple(np.subtract(im.shape, filter.shape) + 1)
    strd = np.lib.stride_tricks.as_strided
    subM = strd(im, shape = s, strides = im.strides * 2)
    return np.einsum('ij,ijkl->kl', filter, subM)

def denoise_img(img):
    # expects an image of (H, W, C) with values from 0 to 1
    img_out = np.zeros(img.shape)
    
    for x in range(img.shape[2]):
        img_out[:,:,x] = anisodiff(img[:,:,x], 50, 80, 0.02)
        # sharpen per channel output with modified laplacian filter
        kernel = np.array([[-0.5,-1,-0.5], [-1,7,-1], [-0.5,-1,-0.5]])
        img_out[:,:,x] = filter_image(img_out[:,:,x], kernel)

    return np.clip(img_out, 0., 1.)
