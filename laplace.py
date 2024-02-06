# --
# lablace filtering

import numpy as np
from skimage.util.shape import view_as_windows
from scipy.signal import convolve2d


def my_convolve2d(img, k, pad_mode='symmetric'):
  """
  my convolution in 2d
  """

  # shape
  m, n = k.shape

  # image padding
  img_pad = np.pad(img, (m//2, n//2), mode=pad_mode)

  # parts
  img_win = view_as_windows(img_pad, window_shape=k.shape, step=(1, 1))

  # einsum
  img_conv = np.einsum('mnjk,jk->mn', img_win, k)

  # reshape
  img_conv = img_conv.reshape(img.shape)

  return img_conv


def laplace_filter(img):

  # define kernel
  k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

  # convolve
  #img_l = convolve2d(img, k, mode='full', boundary='symm', fillvalue=0)
  #img_l = convolve2d(img, k, mode='same', boundary='fill', fillvalue=0)
  img_l = convolve2d(img, k, mode='same', boundary='symm', fillvalue=0)

  # my conv
  #img_l = my_convolve2d(img, k, pad_mode='symmetric')

  return img_l


if __name__ == '__main__':
  """
  main
  """

  # get image
  img = np.arange(9*9).reshape(9, 9)
  print(img), print(img.shape)

  # filter image
  img_l = laplace_filter(img)
  print(img_l),print(img_l.shape)