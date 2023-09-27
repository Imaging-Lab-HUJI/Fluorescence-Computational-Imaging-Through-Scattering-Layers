import os
import torch
import numpy as np
from torch.fft import *

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

# Retrieve Object from Matrix
def RtoO(x, imsize, real=False):
    # Decide on the type of inverse FFT to use based on the `real` flag
    i = irfft2 if real else ifft2

    # Find Original measurements sizes
    M, Nx = x.shape[1], imsize[0]
    Ny = x.shape[0] // Nx

    # Reshape reflection matrix back to measurements
    T = torch.permute(x.reshape(Ny, Nx, M), [2, 1, 0]).cpu()

    # Compute object
    return torch.sqrt(torch.mean(i(T).abs() ** 2, 0)).reshape(*imsize)


def CTR_CLASS(R: torch.Tensor,num_iters = 100,save_path=None,save_name='0',imsize = None,real=False,device = None):
    """
        Estimates the Optical Transfer Function (OTF) using the CTR-CLASS algorithm.

        Parameters:
            R (torch.Tensor): The input complex-valued Reflection Matrix in the fourier domain, as a torch.Tensor.
            num_iters (int, optional): Number of iterations for the algorithm (default is 100).
            save_path (str, optional): Path to save intermediate results (default is the current working directory).
            save_name (str, optional): A name prefix for saved files (default is '0').
            imsize (list, optional): A list containing two integers for the output image size (default is None).
            real (bool, optional): If True, assume the input is a result of rfft2 of a real-valued input  (default is False).
            device (torch.device, optional): Device to perform computations (default is GPU if available, else CPU).

        Returns:
            tuple: A tuple containing the following elements:
                - R (torch.Tensor): The corrected and updated Reflection Matrix after the iterations.
                - O_est (torch.Tensor): The estimated object.
                - phi_tot (torch.Tensor): The correction phase mask.
                - OTF (torch.Tensor): The absolute value of the Optical Transfer Function (OTF).
        """
    # Determine which device to use for computations
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    R = R.to(device)

    # Set default save path if not provided, to current folder
    if save_path is None:
        save_path = os.getcwd()

    # Matrix and output image sizes
    N,M = R.shape
    if imsize is None:
        imsize = 2*[int(N**.5)]


    O0 = RtoO(R,imsize,real=real) # Estimate object before starting
    np.save(os.path.join(save_path,f'Oest0_{save_name}.npy'), O0)

    # Set initial phase mask to 0
    phi_tot = torch.ones(N, dtype=torch.complex64,device=device)

    # Iterative CLASS algorithm
    for k in range(1,1+num_iters):
        # Compute OTF estimation
        Cnv = fftconv(torch.conj(R.flipud()), R.fliplr())[:, M - 1]
        temp = torch.roll(fftcorr(R.flipud(), Cnv.unsqueeze(1).conj()).flipud(), 1, 0)[:N, :M]
        OTF = torch.mean((R.conj()) * temp, 1)

        phi = torch.exp(1j*(OTF.angle())) # Take only phases

        # Update phase mask and correct matrix
        phi_tot *= phi
        R = (phi * R.T).T

        if k % 10 == 0:
            print(f'Iteration {k}/{num_iters}')

    # Final estimation of the object and other outputs
    OTF = OTF.cpu().abs()
    O_est = RtoO(R,imsize,real=real)
    np.save(os.path.join(save_path, f'Oest_{save_name}.npy'), O_est)
    np.save(os.path.join(save_path, f'OTF_{save_name}.npy'), OTF)

    return R.cpu(), O_est, phi_tot.cpu(), OTF
