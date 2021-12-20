import numpy as np
import numba as numbat
import time

@numbat.njit
def sharpen_image(new_im,old_im,algo):
    for i in np.arange(1,old_im.shape[0]-1):
        for j in range(1,old_im.shape[1]-1):
            tmp=old_im[i-1,j-1]*256+old_im[i-1,j]*128+old_im[i-1,j+1]*64
            tmp=tmp+old_im[i,j-1]*32+old_im[i,j]*16+old_im[i,j+1]*8
            tmp=tmp+old_im[i+1,j-1]*4+old_im[i+1,j]*2+old_im[i+1,j+1]
            new_im[i,j]=algo[tmp]
if True:
    algo=np.loadtxt('input_dec20_algo.txt',delimiter=' ',dtype='int')
    imstart=np.loadtxt('input_dec20_image.txt',delimiter=' ',dtype='int')
else:
    algo=np.loadtxt('input_algo.txt',delimiter=' ',dtype='int')
    imstart=np.loadtxt('input_im.txt',delimiter=' ',dtype='int')

niter=50
pad=2*niter+3
myim=np.zeros([imstart.shape[0]+pad*2,imstart.shape[1]+2*pad],dtype='int')
myim[pad:-pad,pad:-pad]=imstart

new_im=0*myim
t1=time.time()
for iter in range(niter):
    sharpen_image(new_im,myim,algo)
    myim=new_im
    new_im=0*new_im
t2=time.time()
print('npix is ',np.sum(myim[iter:-iter,iter:-iter]))
print('took ',t2-t1,' seconds to run.')
