import tensorflow as tf
from PIL import Image
import math
import numpy as np
import cv2

def int_shape(x):
    if str(x.get_shape()[0]) != '?':
        return list(map(int, x.get_shape()))
    return [-1]+list(map(int, x.get_shape()[1:]))

def adjust_dynamic_range(data, drange_in, drange_out):
    if drange_in != drange_out:
        scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (
                    np.float32(drange_in[1]) - np.float32(drange_in[0]))
        bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
        data = data * scale + bias
    return data

def write_images(images, filename, drange, mode, format='png', text=None):
    sq = math.floor(math.sqrt(len(images)))
    assert sq ** 2 == len(images)
    sq = int(sq)
    image_rows = [np.concatenate(images[i:i + sq], axis=0)
                  for i in range(0, len(images), sq)]
    tiled_image = np.concatenate(image_rows, axis=1)
    tiled_image = adjust_dynamic_range(tiled_image, drange, [0, 255])
    tiled_image = np.clip(tiled_image, 0, 255)
    tiled_image = np.uint8(tiled_image)
    if text is not None:
        y = 40
        font = cv2.FONT_HERSHEY_SIMPLEX # pylint: disable=E1101
        for i in range(len(text)):
            cv2.putText(tiled_image, text[i], (20, y), font, 1, (255, 255, 255), 2, cv2.LINE_AA) # pylint: disable=E1101
            y += 35
    img = Image.fromarray(tiled_image, mode=mode)
    file_obj = tf.gfile.Open(filename, 'w')
    img.save(file_obj, format=format)

class RestoreParametersAverageValues(tf.train.SessionRunHook):
   """
   Replace parameters with their moving averages.
   This operation should be executed only once, and before any inference.
   """
   def __init__(self, ema):
       """
       :param ema:         tf.train.ExponentialMovingAverage
       """
       super(RestoreParametersAverageValues, self).__init__()
       self._ema = ema
       self._restore_ops = None

   def begin(self):
       """ Create restoring operations before the graph been finalized. """
       ema_variables = tf.moving_average_variables()
       self._restore_ops = [tf.assign(x, self._ema.average(x)) for x in ema_variables]

   def after_create_session(self, session, coord):
       """ Restore the parameters right after the session been created. """
       session.run(self._restore_ops)

def track_params_averages(scope):
    ema = tf.train.ExponentialMovingAverage(0.999)    
    params_averages_op = ema.apply(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope))
    return ema, params_averages_op
