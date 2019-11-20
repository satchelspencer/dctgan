## working on 2070
from __future__ import division, print_function, unicode_literals
import tensorflow as tf
import glob
import numpy as np
import os
import PIL
from PIL import Image
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.preprocessing import image
from tensorflow import keras
import time
import os
import h5py
from dct import bsize, dctize, imgize, count

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# tf.enable_eager_execution(config)

size=512
dctsize = int(size/bsize)
latent_dim = 100
latent_shape = (latent_dim,)
channel_mult = 16
BATCH_SIZE = 64
RAW = True

regscale = 0.0001
def ortho_reg(w) :
  """ Reshaping the matrxi in to 2D tensor for enforcing orthogonality"""
  _, _, _, c = w.get_shape().as_list()
  w = tf.reshape(w, [-1, c])
  """ Declaring a Identity Tensor of appropriate size"""
  identity = tf.eye(c)
  """ Regularizer Wt*W - I """
  w_transpose = tf.transpose(w)
  w_mul = tf.matmul(w_transpose, w)
  reg = tf.subtract(w_mul, identity)
  """Calculating the Loss Obtained"""
  ortho_loss = tf.nn.l2_loss(reg)
  return scale * ortho_loss

class CondBatchNorm(layers.Layer):
  def __init__(self):
    super(CondBatchNorm, self).__init__()
    self.decay = 0.9
    self.epsilon = 1e-05

  def build(self, ishape):
    channels = ishape[0][-1]
    self.densebeta = layers.Dense(channels)
    self.densegamma = layers.Dense(channels)
    self.testMean = tf.Variable(name='testMean', initial_value=tf.zeros((channels,)), dtype='float32', trainable=False)
    self.testVar = tf.Variable(name='testVar', initial_value=tf.ones((channels,)), dtype='float32', trainable=False)
   
  def call(self, x, training=None):
    channels = x[0].get_shape().as_list()[-1]
    beta = self.densebeta(x[1])
    gamma = self.densegamma(x[1])
    beta = tf.reshape(beta, shape=[-1, 1, 1, channels])
    gamma = tf.reshape(gamma, shape=[-1, 1, 1, channels])
    #return x[0]
    if training:
      batch_mean, batch_var = tf.nn.moments(x[0], [0, 1, 2])
      self.testMean.assign(self.testMean * self.decay + batch_mean * (1 - self.decay))
      self.testVar.assign(self.testVar * self.decay + batch_var * (1 - self.decay))
      return tf.nn.batch_normalization(x[0], batch_mean, batch_var, beta, gamma, self.epsilon)
    else: 
      return tf.nn.batch_normalization(x[0], self.testMean, self.testVar, beta, gamma, self.epsilon)

def gen_block(channels):
  def layer(inp, z):
    x = CondBatchNorm()([inp, z])
    x = layers.LeakyReLU()(x)
    x = layers.UpSampling2D((2,2))(x)
    x = layers.Conv2D(channels, (3,3), padding='same', kernel_regularizer=ortho_reg)(x)
    x = CondBatchNorm()([x, z])
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(channels, (3,3), padding='same', kernel_regularizer=ortho_reg)(x)

    skip = layers.UpSampling2D((2,2))(inp)
    skip = layers.Conv2D(channels, (3,3), padding='same', kernel_regularizer=ortho_reg)(skip)
    return x + skip
  return layer

def hw_flatten(x) :
  s = tf.shape(x)
  return tf.reshape(x, shape=[s[0], -1, s[-1]])

class SelfAttention(layers.Layer):
  def __init__(self):
    super(SelfAttention, self).__init__()
    self.gamma = self.add_weight(
      name='sa_gamma',
      shape=(1,),
      initializer='zeros',
      dtype='float32',
      trainable=True
    )

  def build(self, ishape):
    channels = ishape[-1]
    self.fconv = layers.Conv2D(channels // 8, (1,1), strides=(1,1), kernel_regularizer=ortho_reg)
    self.gconv = layers.Conv2D(channels // 8, (1,1), strides=(1,1), kernel_regularizer=ortho_reg)
    self.hconv = layers.Conv2D(channels, (1,1), strides=(1,1), kernel_regularizer=ortho_reg)
    self.oconv = layers.Conv2D(channels, (1,1), strides=(1,1), kernel_regularizer=ortho_reg)
   
  def call(self, x):
    channels = x.get_shape().as_list()[-1]
    f = self.fconv(x) # [bs, h, w, c']
    g = self.gconv(x) # [bs, h, w, c']
    h = self.hconv(x) # [bs, h, w, c]

    # N = h * w
    s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True) # # [bs, N, N]

    beta = tf.nn.softmax(s)  # attention map

    o = tf.matmul(beta, hw_flatten(h)) # [bs, N, C]

    o = tf.reshape(o, shape=tf.shape(x)) # [bs, h, w, C]
    o = self.oconv(o)
    x =  self.gamma * o + x

    return x

def make_generator_model():
  inp = Input(shape=latent_shape)
  firstc = channel_mult*16
  x = layers.Dense(8*8*firstc)(inp)
  x = layers.BatchNormalization()(x)
  x = layers.LeakyReLU()(x)
  x = layers.Reshape((8, 8, firstc))(x)

  x = gen_block(firstc)(x, inp)
  x = gen_block(channel_mult*8)(x, inp)
  x = SelfAttention()(x)
  x = gen_block(channel_mult*4)(x, inp)
  # x = gen_block(channel_mult*4)(x, inp)
  # x = gen_block(channel_mult*2)(x, inp)

  x = layers.BatchNormalization()(x)
  x = layers.LeakyReLU()(x)
  out = layers.Conv2D(3*count, (3,3), padding='same', activation='tanh', kernel_regularizer=ortho_reg)(x)

  return Model(inp, out)

class BatchStd(layers.Layer):
  def __init__(self, group_size=4):
    super(BatchStd, self).__init__()
    self.group_size = group_size

  def call(self, x):
    group_size = tf.minimum(self.group_size, tf.shape(x)[0])# Minibatch must be divisible by (or smaller than) group_size.
    s = x.shape                                             # [NCHW]  Input shape.
    y = tf.reshape(x, [group_size, -1, s[1], s[2], s[3]])   # [GMCHW] Split minibatch into M groups of size G.
    y = tf.cast(y, tf.float32)                              # [GMCHW] Cast to FP32.
    y -= tf.reduce_mean(y, axis=0, keepdims=True)           # [GMCHW] Subtract mean over group.
    y = tf.reduce_mean(tf.square(y), axis=0)                # [MCHW]  Calc variance over group.
    y = tf.sqrt(y + 1e-8)                                   # [MCHW]  Calc stddev over group.
    y = tf.reduce_mean(y, axis=[1,2,3], keepdims=True)      # [M111]  Take average over fmaps and pixels.
    y = tf.cast(y, x.dtype)                                 # [M111]  Cast back to original data type.
    y = tf.tile(y, [group_size, 1, s[2], s[3]])             # [N1HW]  Replicate over group and pixels.
    return tf.concat([x, y], axis=1) 

def dsc_block(channels):
  def layer(inp):
    x = layers.LeakyReLU()(inp)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(channels, (3,3), strides=(1, 1), padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(channels, (3,3), strides=(1, 1), padding='same')(x)
    x = layers.AvgPool2D()(x)

    skip = layers.Conv2D(channels, (1,1), strides=(1, 1), padding='same')(inp)
    skip = layers.AvgPool2D()(skip)
    return x + skip
  return layer

def make_discriminator_model():
  inp = Input(shape=(dctsize,dctsize,3*count))
  x = dsc_block(channel_mult*2)(inp)
  x = dsc_block(channel_mult*4)(x)
  # x = dsc_block(channel_mult*8)(x)
  x = SelfAttention()(x)
  x = dsc_block(channel_mult*8)(x)

  x = BatchStd(group_size=4)(x)

  x = layers.LeakyReLU()(x)
  x = layers.Flatten()(x)
  out = layers.Dense(1)(x)
  return Model(inp, out)

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

generator = make_generator_model()
#generator.summary()

discriminator = make_discriminator_model()
#discriminator.summary()

#generator.load_weights('./33800g.h5')
#discriminator.load_weights('./33800d.h5')

print('getting training data...')
frames_dir = "./raw" if RAW else "./frames-512"#"/Users/satchel/dev/pvsn-automataa/video_canvas/frames/frames-128"

if RAW:
  frames = np.array(
      [
          top + os.sep + f
          for top, dirs, files in os.walk(frames_dir)
          for f in files
          if (f[0] != '.' and f[-4:] == "hdf5")
      ]
  )
else:
  frames = np.array(
      [
          top + os.sep + f
          for top, dirs, files in os.walk(frames_dir)
          for f in files
          if (f[0] != '.' and f[-3:] == "jpg" and int(f[:5]) > 120 and int(f[:5]) < 480)
      ]
  )
print('found', len(frames), 'frames')


@tf.function
def train_step(images):
  noise = tf.random.normal([BATCH_SIZE, latent_dim])

  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    generated_images = generator(noise, training=True)

    real_output = discriminator(images, training=True)
    fake_output = discriminator(generated_images, training=True)

    gen_loss = generator_loss(fake_output)
    disc_loss = discriminator_loss(real_output, fake_output)

  gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
  gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
  return (gen_loss, disc_loss)

if __name__ == '__main__':
  for epoch in range(50000):
    for i in range(100):
      idx = np.random.randint(0, len(frames), BATCH_SIZE)
      img_paths = frames[idx]
      
      if RAW:
        batch = []
        for imgpath in img_paths:
          f = h5py.File(imgpath, 'r')
          batch.append(np.array(f['dct'], dtype=np.float32))
          f.close()
        batch = np.array(batch)
      else:
         batch = np.array(
           [
             dctize(np.asarray(image.load_img(imgpath, target_size=(size, size)), dtype=np.float32))
             for imgpath in img_paths
           ]
         )

      gen_loss, disc_loss = train_step(batch)
      if i == 0:
        print(float(gen_loss), float(disc_loss))

    # save images
    test_input = tf.random.normal([1, latent_dim])
    predictions = generator(test_input, training=False)
    #sgen_imgs = (0.5 * predictions + 0.5   ) * 255
    dct = np.array(predictions)[0]

    # print(np.min(dct, axis=(0,1)))
    # print(np.max(dct, axis=(0,1))-np.min(dct, axis=(0,1)))
    #print(np.array(predictions)[0][0,0])
    Image.fromarray(np.uint8(imgize(dct))).save("./img/%s.jpg" % epoch)

    # Save the model
    if (epoch) % 100 == 0:
      generator.save_weights('weights/%dg.h5' % epoch)
      discriminator.save_weights('weights/%dd.h5' % epoch)
