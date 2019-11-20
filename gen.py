from __future__ import division, print_function, unicode_literals
import glob
import numpy as np
import os
from PIL import Image
import time
import os
from dct import bsize, dctize, imgize, count
import h5py

frames_dir = "./frames-512"
out_dir = "./raw"

i = 0
for top, dirs, files in os.walk(frames_dir):
  for f in files:
    path = top + os.sep + f
    if f[0] != '.' and f[-3:] == "jpg" and int(f[:5]) > 250 and int(f[:5]) < 350:
      im = np.array(Image.open(path)).astype(float)
      dct = dctize(im)
      h = h5py.File(out_dir + os.sep + str(i) + ".hdf5", "w")
      h.create_dataset("dct", data=dct, dtype=np.float32)
      h.close()
      i = i+1
      if i % 100 == 0:
        print(i)
