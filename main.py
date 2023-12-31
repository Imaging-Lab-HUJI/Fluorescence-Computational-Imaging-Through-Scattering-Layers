import numpy as np
import torch
from ICLASS import I_CLASS
from torchvision.transforms import CenterCrop
from visualize import showResults
import os
from mat73 import loadmat

def mkdir(pth):
    if not os.path.isdir(pth):
        os.mkdir(pth)


## Reads .mat file of measurements, inserts into a Tensor sized [M,Nx,Ny]
## Can also read Ground Truths
def readMAT(matFile,matFileGT = None):

    # Read Ground Truth
    gt = matFileGT
    if gt is not None:
        gt = loadmat(gt)['I_cam']
    im = loadmat(matFile)['I_cam']

    # Read measurements and arrange it into right dimensions
    Icam = torch.permute(torch.from_numpy(im).float(), [2, 0, 1])

    return Icam, gt

def runCLASS(Icam,num_iters=100,cut = None,keepAspect = False,saveName='0',savePath=None):
    """
    Run the I-CLASS algorithm on input measurements.

    Parameters:
        Icam (torch.Tensor): Input data images as a MxHxW torch.Tensor.
        num_iters (int, optional): Number of iterations for the I-CLASS algorithm (default is 100).
        cut (int, optional): Fourier Domain cut size (default is no cutting).
        keepAspect (bool, optional): If True, maintain the aspect ratio when cutting (default is False).
        saveName (str, optional): A name prefix for saved files (default is '0').
        savePath (str, optional): Path to save intermediate results (default is current folder).

     Returns:
        tuple: A tuple containing the following elements:
            - R (torch.Tensor): The corrected and updated Reflection Matrix after the I-CLASS algorithm.
            - O_est (torch.Tensor): The estimated object.
            - phi_tot (torch.Tensor): The estimated correction phase mask.
            - OTF (torch.Tensor): The estimated absolute value of the Optical Transfer Function (OTF).
    """

    # Set output size
    if cut is None or cut < 0:
        imsize = Icam.shape[-2:]
    else:
        imsize = 2*[cut]
        if keepAspect:
            x, y = Icam.shape[:2]
            imsize[0] = x * cut // y
        C = CenterCrop(imsize)
        # Foruier Transform and Cutting (Resize with Fourier LPF interpolation)
        Icam = torch.fft.ifft2(torch.fft.ifftshift(C(torch.fft.fftshift(torch.fft.fft2(Icam))))).real

    ## Pre-processing
    Icam -= Icam.mean(0) # Mean Reduction
    Icam_fft = torch.fft.fftshift(torch.fft.fft2(Icam)) # Fourier Transform

    # Reshape into Reflection Matrix
    R = torch.permute(Icam_fft, [2, 1, 0]).reshape(Icam_fft.shape[1]*Icam_fft.shape[2], -1)

    ## Run I-CLASS
    return I_CLASS(R, num_iters=num_iters,save_path=savePath, save_name=saveName, imsize=imsize)

## Parameters
DATA_PATH = os.path.join('DATA','08-Aug-2023') # Enter Here the data path
meas_idx, ground_truth_idx = 4 ,3 # Set measurements and ground truth index (if ground truth doesn't exits, set ground_truth_idx = -1 or ground_truth_idx = None)
cut = -1 # Set Fourier Domain cut size for (Optional, if not wanted set cut = -1 or cut = None)
CLASS_iterations = 250 # Set I-CLASS number of iterations


## Create Results Folder
SAVE_PATH = os.path.join(DATA_PATH, 'Results')
mkdir(SAVE_PATH)

## Read Files
gt_path = None
if ground_truth_idx is not None and ground_truth_idx >= 0:
    gt_path = os.path.join(DATA_PATH,f'Run_{ground_truth_idx}.mat')
Icam,gt = readMAT(os.path.join(DATA_PATH,f'Run_{meas_idx}.mat'),gt_path)
# Save Ground Truth
if gt is not None:
    np.save(os.path.join(SAVE_PATH,f'gt_{meas_idx}.npy'),gt)


## Run algorithm
T, O_est, phi_tot,OTF = runCLASS(Icam, num_iters=CLASS_iterations, cut=cut, saveName=f'{meas_idx}',savePath=SAVE_PATH)
torch.save(phi_tot, os.path.join(SAVE_PATH,f'phi_{meas_idx}.trc'))

## Visualize Results
showResults(DATA_PATH,meas_idx=meas_idx)
