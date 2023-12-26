import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.widgets import Slider

import torch
import numpy as np
from torchvision.transforms import CenterCrop
from scipy.ndimage import shift
from torch.fft import *

pi = np.pi

## Create ColorMap

# Define the colors for the colormap in RGB format
colors = [
    (0, 0, 0),    # Black
    (0, 0.2, 0),  # Dark Green
    (0, 0.5, 0),  # Green
    (0, 0.8, 0),  # Bright Green
    (0.7, 1, 0),  # Light Green-Yellow
    (1, 1, 1)     # White
]
# Define the positions for the colors in the colormap (0 to 1)
positions = [0, 0.2, 0.4, 0.6, 0.8, 1]
# Create the colormap using LinearSegmentedColormap
new_cmap = LinearSegmentedColormap.from_list('greenish_hot', list(zip(positions, colors)))


# Define utility functions like crop_center, CC, center_image, shift_cross_correlation, deconv, and nrm.
# These functions are used for various image processing tasks, including deconvolution and centering.

def crop_center(img, cropx, cropy):
    y, x = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty + cropy, startx:startx + cropx]
def CC(x,y):
    q = fft2(x) * torch.conj(fft2(y))
    return torch.fft.fftshift(torch.fft.ifft2(q/torch.abs(q)))
def center_image(image,COM=False):
    if COM:
        # Calculate the center of mass
        center_of_mass_x = np.sum(np.arange(image.shape[1]) * image.sum(axis=0)) / image.sum()
        center_of_mass_y = np.sum(np.arange(image.shape[0]) * image.sum(axis=1)) / image.sum()

        # Calculate the desired center
        desired_center_x = image.shape[1] / 2
        desired_center_y = image.shape[0] / 2

        # Determine the shift required
        shift_x = desired_center_x - center_of_mass_x
        shift_y = desired_center_y - center_of_mass_y

    else:
        max_pixel_y, max_pixel_x = np.unravel_index(image.argmax(), image.shape)

        # Calculate the desired center
        desired_center_x = image.shape[1] / 2
        desired_center_y = image.shape[0] / 2

        # Determine the shift required
        shift_x = desired_center_x - max_pixel_x
        shift_y = desired_center_y - max_pixel_y
    shifted_image = shift(image, (shift_y, shift_x),mode='wrap')
    return shifted_image
def shift_cross_correlation(dest, src):
    shp = np.array(dest.shape[-2:])
    q = CC(dest,src).abs()
    mxidx = np.array(np.unravel_index(np.argmax(q), shp))
    max_loc = shp // 2 - mxidx
    return shift(src.abs().cpu().numpy(),-np.array(max_loc),mode='wrap')
def deconv(Ok,kernel,sig):
    return nrm(torch.fft.ifft2(Ok / torch.fft.ifftshift(kernel + sig)).abs())
nrm = lambda x:x/x.abs().max()

def showResults(data_path,meas_idx):
    # Function to visualize results from CTR-CLASS
    SAVE_PATH = os.path.join(data_path,'Results')

    # Load Initial Object, FInal Iteration Object,  correction phase, and |OTF| estimation
    O0 = np.load(os.path.join(SAVE_PATH,f'Oest0_{meas_idx}.npy'))
    O = np.load(os.path.join(SAVE_PATH,f'Oest_{meas_idx}.npy'))
    phi = torch.load(os.path.join(SAVE_PATH,f'phi_{meas_idx}.trc'))
    N = O0.shape[0] # Get Size of Object
    MTF = nrm(torch.from_numpy(np.load(os.path.join(SAVE_PATH,f'MTF_{meas_idx}.npy'))).resize(*(2*[N])).T.abs())

    # Load Ground Truth if exists
    gt_path = os.path.join(SAVE_PATH,f'gt_{meas_idx}.npy')
    gt = None
    if os.path.exists(gt_path):
        gt = center_image(np.load(gt_path).mean(2), COM=True)
        gt = torch.fft.ifft2(
            torch.fft.ifftshift(CenterCrop(N)(torch.fft.fftshift(torch.fft.fft2(torch.from_numpy(gt)))))).abs().numpy()

    # Shift Object to correlate to Ground Truth)
    O = nrm(torch.from_numpy(shift_cross_correlation(torch.from_numpy(gt), torch.from_numpy(O))))
    Ok = nrm(torch.fft.fft2(O))

    lbls = ['Inital Object','CLASS Final','GT','Deconvolution']
    # Setting up the figure and axes
    fig, axarr = plt.subplots(2, 2)
    # Initial s value and initial image data
    s = 0.00001

    # Displaying the images in 1x2 format
    imgs = []
    for (i,ax),im in zip(enumerate(axarr.ravel()), [O0,O,gt,deconv(Ok,MTF,s)]):
        img = ax.imshow(im, cmap=new_cmap)
        ax.set_title(lbls[i])
        imgs.append(img)

    # Adding the Slider for the parameter s
    axcolor = 'lightgoldenrodyellow'
    ax_slider = plt.axes([0.05, 0.01, 0.85, 0.03], facecolor=axcolor)
    s_slider = Slider(ax_slider, r'$\sigma$', s, 0.5, valinit=s)

    # Update function for the slider
    def update(val):
        imgs[-1].set_data(deconv(Ok,MTF,s_slider.val))
        fig.canvas.draw_idle()
    # Attach the update function to the slider
    s_slider.on_changed(update)
    plt.show()
