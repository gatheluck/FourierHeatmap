import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
sys.path.append(base)

import numpy as np
import torch
import torchvision

def generate_fourier_base(h:int, w:int, h_index:int, w_index:int):
    """
    return normalized 2D fourier base function.
    FFT(fourier_base)(=spectrum_matrix) should have only two non zero element at (-h_index, -w_index) and (+h_index, +w_index).
    for detail, please refer section 2 (Preliminaries) of original paper: https://arxiv.org/abs/1906.08988 .
    
    Args
    - h: height of output image.
    - w: width  of output image. 
    - h_index: spectrum index of height dimension. -np.floor(h/2) <= h_index <= +np.floor(h/2) should be satisfied.
    - w_index: spectrum index of width  dimension. -np.floor(w/2) <= w_index <= +np.floor(w/2) should be satisfied.
    
    Return
    - fourier_base: normalized 2D fourier base function
    """

    assert h>=1 and w>=1
    assert abs(h_index) <= np.floor(h/2) and abs(w_index) <= np.floor(w/2)

    h_center_index = int(np.floor(h/2)) 
    w_center_index = int(np.floor(w/2))

    spectrum_matrix = torch.zeros(h,w)
    spectrum_matrix[h_center_index+h_index, w_center_index+w_index] = 1.0
    if (h_center_index-h_index) < h and (w_center_index-w_index) < w:
        spectrum_matrix[h_center_index-h_index, w_center_index-w_index] = 1.0

    spectrum_matrix = spectrum_matrix.numpy()
    spectrum_matrix = np.fft.ifftshift(spectrum_matrix) # swap qadrant (low-freq centered to high-freq centered)

    fourier_base = torch.from_numpy(np.fft.ifft2(spectrum_matrix).real).float()
    fourier_base /= fourier_base.norm()

    return fourier_base

if __name__ == "__main__":
    def generate_fourier_basis(h:int, w:int, file_path):
        x_list=[]
        for i in range(-int(np.floor(h/2)), h-int(np.floor(h/2))):
            for j in range(-int(np.floor(w/2)), w-int(np.floor(w/2))):
                fourier_base = generate_fourier_base(h,w,i,j)

                x = torch.zeros(1,1,1)
                x += 0.5
                x = x.repeat(3,h,w)

                x[0,:,:] += fourier_base * 10.0
                x[1,:,:] += fourier_base * 10.0  
                x[2,:,:] += fourier_base * 10.0

                x = torch.clamp(x, min=0.0, max=1.0)
                x_list.append(x)

        torchvision.utils.save_image(x_list, file_path, nrow=w)
    
    os.makedirs('../logs', exist_ok=True)
    generate_fourier_basis(32,32,'../logs/fourier_base_32x32.png')