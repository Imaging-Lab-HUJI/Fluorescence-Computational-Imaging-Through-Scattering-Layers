import torch
from torch.fft import *
import os

pi = torch.pi

# Convolution using FFT
def fftconv(A, B, mode='same'):
    if mode in ['Full','full','pad','Pad']:
        # Calculate the size of the padded output
        out_size = A.size(-2) + B.size(-2) - 1, A.size(-1) + B.size(-1) - 1

        # Perform 2D FFT on both input matrices and multiply them
        f = fft2(A, s=out_size)
        f *= fft2(B, s=out_size)
    else:
        f = fft2(A)
        f *= fft2(B)
    # Perform inverse 2D FFT on the result and normalize
    return (ifft2(f)) / (torch.numel(f))


# Cross-Correlation using FFT
def fftcorr(A, B, mode='same'):
    if mode in ['Full','full','pad','Pad']:
        # Calculate the size of the padded output
        out_size = A.size(-2) + B.size(-2) - 1, A.size(-1) + B.size(-1) - 1

        # Perform 2D FFT on both input matrices and multiply them (with conjugate for B)
        f = fft2(A, s=out_size)
        f *= (fft2(B, s=out_size).conj())
    else:
        f = fft2(A)
        f *= (fft2(B).conj())

    # Perform inverse 2D FFT on the result and normalize
    return ifft2(f) / (torch.numel(f))

def mkdir(pth):
    if not os.path.isdir(pth):
        os.mkdir(pth)

