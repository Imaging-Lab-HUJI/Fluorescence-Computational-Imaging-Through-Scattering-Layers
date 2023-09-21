import numpy as np
import torch
from CTRCLASS import CTR_CLASS
import os
import matplotlib.pyplot as plt
from torchvision.transforms import CenterCrop
from utils import gauss2D, mkdir
from visualize import showResults
import os
wavelength = np.float64(532e-3) # [μm]
from mat73 import loadmat

## Reads .mat file of measurements, inserts into a Tensor sized [M,Nx,Ny]
## Can also read Ground Truths
def readMAT(matFile,matFileGT = None):
    # Read Ground Truth
    gt = matFileGT
    if gt is not None:
        gt = loadmat(gt)['I_cam']
    im = loadmat(matFile)['I_cam']
    # for j in range(im.shape[-1]):
    #     im[:,:,j] = medfilt2d(im[:,:,j],5)

    # Read measurements and arrange it into right dimensions
    Icam = torch.permute(torch.from_numpy(im).float(), [2, 0, 1])

    return Icam, gt

def runCLASS(Icam,num_iters=100,cut = 500,keepAspect = False,saveName='Inchoerent',savePath=None):
    # Set output size
    imsize = 2*[cut]
    if keepAspect:
        x, y = Icam.shape[:2]
        imsize[0] = x * cut // y
    C = CenterCrop(imsize)



    # Icam = LowPass(Icam,10).abs()

    # Icam *= torch.outer(*[torch.hann_window(s) for s in Icam.shape[1:]])
    # plt.imshow((Icam[0]))
    # plt.show()
    # plt.imshow(torch.fft.fftshift(torch.fft.fft2(Icam[0])).abs())
    # plt.show()
    # Set mean to 0
    # Icam /= Icam.max()
    # Icam -= Icam.mean(dim=(-2, -1), keepdim=True)
    Icam -= Icam.mean(0)
    # Foruier Transform and Cutting (Resize with Fourier LPF interpolation)

    # plt.imshow(torch.log(torch.fft.fftshift(torch.fft.fft2(Icam[0])).abs()))
    # plt.show()
    Icam = torch.fft.ifft2(torch.fft.ifftshift(C(torch.fft.fftshift(torch.fft.fft2(Icam))))).real
    Icam_fft = torch.fft.fftshift(torch.fft.fft2(Icam))

    # Icam_fft = torch.fft.rfft2(Icam)

    # Reshape into Reflection Matrix
    N = Icam_fft.shape[1]*Icam_fft.shape[2]
    T = torch.permute(Icam_fft, [2, 1, 0]).reshape(N, -1)

    T, O_est, phi_tot,OTF = CTR_CLASS(T,save_every=20000000, num_iters=num_iters,save_path=savePath, save_name=saveName, imsize=imsize,real=False)

    return T.cpu(), O_est.cpu(), phi_tot.cpu(),OTF

DATA_PATH = '08-Aug-2023'
DATA_PATH = '05-Sep-2023'

SAVE_PATH = os.path.join(DATA_PATH,'Results')
mkdir(SAVE_PATH)
rfftToFull = lambda x: torch.fft.fft2(torch.fft.irfft2(x))
for i in [6]:
    torch.cuda.empty_cache()
    # Read Files
    meas_path, gt_path = os.path.join(DATA_PATH,f'Run_{i}.mat'), os.path.join(DATA_PATH,f'Run_{5}.mat')
    Icam,gt = readMAT(meas_path,gt_path)

    print(f'shape is {Icam.shape}')
    # plt.imshow((Icam[0]).log10())
    # plt.title(f'i = {i}')
    # plt.show()
    # continue
    # Icam = torch.nn.functional.pad(Icam[:,450:650,800:1000],(50,50,50,50))
    # Icam = Icam[:,250:1500,250:1500]
    # Icam = Icam[:,550:1500,750:]
    # Icam /= Icam.max()

    # Icam = torch.from_numpy(medfilt2d(Icam.numpy()))
    # Icam -= Icam.mean(0)
    # for i in range(Icam.shape[0]):
    #     plt.imshow((Icam[i]))
        # plt.clim(0,1)
    #     plt.show()
    #     plt.waitforbuttonpress(0.0001)
    # if i == 4:
    #     Icam = Icam[:,100:,100:]
    # if i == 3:
    #     Icam = Icam[:,300:1700,300:1700]
    # for i in range(Icam.shape[0]):
    #     plt.imshow((Icam[i]))
    #     plt.waitforbuttonpress(0.0001)
    # plt.imshow(torch.log(Icam[0]))
    # plt.show()
    if gt is not None:
        np.save(os.path.join(SAVE_PATH,f'gt_{i}.npy'),gt)
    cut = 900
    with torch.no_grad():
        T, O_est, phi_tot,OTF = runCLASS(Icam, num_iters=250, cut=cut, saveName=f'm{i}',savePath=SAVE_PATH)

        plt.imshow(O_est)
        plt.show()
    # phi_tot = phi_tot.reshape(cut,cut//2 + 1)
        torch.save(phi_tot, os.path.join(SAVE_PATH,f'phi_{i}.trc'))
    showResults(DATA_PATH,meas_idx=i)