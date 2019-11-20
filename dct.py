import numpy as np
import scipy
import time
from numpy import pi
from numpy import sin
from numpy import zeros
from numpy import r_
from scipy import signal
from scipy import misc 
import PIL
from PIL import Image
import h5py

np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)

bsize = 8
quality = 15/64
count = int(bsize*bsize*quality)
coords = (np.zeros(count).astype(int), np.zeros(count).astype(int))
indicies = np.zeros(count).astype(int)
for n in range(count):
  diag = round(np.sqrt((n+1)*2)-1)
  dstart = (diag*(diag+1))/2
  ddiff = n-dstart
  x = int(ddiff)
  y = int(diag-ddiff)
  coords[0][n] = x
  coords[1][n] = y
  indicies[n] = x*bsize + y

spectralscale = 128

def normBlock(block):
  #return block
  ostart = block[0,0]
  block = (block - nbasis)/nrange
  block[0,0] = (ostart-pbasis)/prange
  return block*2 - 1

def denormBlock(block):
  block = (block +1)/2
  ostart = block[0,0]
  block = block*nrange + nbasis
  block[0,0] = ostart*prange + pbasis
  return block


def cubify(arr, newshape):
  oldshape = np.array(arr.shape)
  repeats = (oldshape / newshape).astype(int)
  tmpshape = np.column_stack([repeats, newshape]).ravel()
  order = np.arange(len(tmpshape))
  order = np.concatenate([order[::2], order[1::2]])
  return arr.reshape(tmpshape).transpose(order).reshape(-1, *newshape)

def uncubify(arr, oldshape):
  N, newshape = arr.shape[0], arr.shape[1:]
  oldshape = np.array(oldshape)    
  repeats = (oldshape / newshape).astype(int)
  tmpshape = np.concatenate([repeats, newshape])
  order = np.arange(len(tmpshape)).reshape(2, -1).ravel(order='F')
  return arr.reshape(tmpshape).transpose(order).reshape(oldshape)


def dctize(im):
  chunks = cubify(im, (bsize, bsize, im.shape[-1]))
  chunks -= 128
  chunks = scipy.fftpack.dct( scipy.fftpack.dct( chunks, axis=1, norm='ortho' ), axis=2, norm='ortho' )
  chunks = chunks.reshape(chunks.shape[0], bsize*bsize, chunks.shape[3])
  chunks = np.take(chunks,indicies, axis=1)
  chunks = chunks.reshape((int(im.shape[0]/bsize), int(im.shape[1]/bsize), chunks.shape[1]*chunks.shape[2]))
  return np.tanh(chunks / spectralscale)

def imgize(chunks):
  chunks = np.arctanh(chunks)
  chunks = chunks.reshape((chunks.shape[0]*chunks.shape[1], count, int(chunks.shape[2]/count)))
  empty = np.zeros((chunks.shape[0], bsize*bsize, chunks.shape[-1]))
  dindicies = np.tile(np.expand_dims(np.tile(indicies, (chunks.shape[0], 1)), 2), chunks.shape[-1])
  np.put_along_axis(empty, dindicies, chunks*spectralscale, 1)
  empty = empty.reshape(empty.shape[0], bsize, bsize, empty.shape[-1])
  empty = scipy.fftpack.idct( scipy.fftpack.idct( empty, axis=2, norm='ortho' ), axis=1, norm='ortho' )
  empty = empty + 128
  side = int(np.sqrt(empty.shape[0]))*bsize
  restored = uncubify(empty, (side, side, empty.shape[-1]))
  restored =  np.clip(restored, 0, 255)
  return restored

if __name__ == '__main__':
  im = np.array(Image.open('/Users/satchel/dev/fuckabouts/tweetdl/outr/d737692c8598cce1.jpg')).astype(float)
  #im = np.clip(np.random.randint(0, 100000, im.shape)/100000*255, 0, 255)

  start = time.time()
  dct = dctize(im)
  print('dct', (time.time()-start), 's')

  rdct = np.tanh(np.random.normal(size=dct.shape, scale=0.5))

  # print(np.min(dct, axis=(0,1)))
  # print(np.max(dct, axis=(0,1))-np.min(dct, axis=(0,1)))

  # f = h5py.File('dct.hdf5', 'r')
  # dct = np.array(f['dct'])

  # f = h5py.File("dct.hdf5", "w")
  # f.create_dataset("dct", data=dct, dtype=np.float32)
  # f.close()

  print(dct.shape)
  dimg = imgize(dct)
  Image.fromarray(np.uint8(dimg)).save('out.png')

  # dctroll = np.moveaxis(dct, -1, 0)
  # dctroll = dctroll * 128 + 128
  # dctroll =  np.clip(dctroll, 0, 255)
  # for i in range(dctroll.shape[0]):
  #   Image.fromarray(np.uint8(dctroll[i])).save('dct%d.png'%i)
