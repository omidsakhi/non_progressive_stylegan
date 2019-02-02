from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from PIL import Image

def parser_2(record):
    features = tf.parse_single_example(record, features={
        'shape': tf.FixedLenFeature([3], tf.int64),
        'data': tf.FixedLenFeature([], tf.string)})
    data = tf.decode_raw(features['data'], tf.uint8)    
    image = tf.reshape(data, features['shape'])    
    image = tf.image.resize_images(image, [256, 256])
    image = tf.cast(image, tf.float32)
    image = tf.clip_by_value(image, 0, 255)
    image = (image * (2.0 / 255.0)) - 1.0    
    return image

def parser_1(serialized_example):  
  features = tf.parse_single_example(
      serialized_example,
      features={
          'image': tf.FixedLenFeature([], tf.string),
          'labels': tf.FixedLenFeature([], tf.string),
      })    
  image = tf.image.decode_jpeg(features['image'])
  image = tf.cast(image, tf.float32) * (2.0 / 255) - 1.0  
  image = tf.reshape(image, [3, 128*128])  
  return image

class InputFunction(object):  
  
  def __init__(self, is_training, noise_dim):
    self.is_training = is_training
    self.noise_dim = noise_dim

  def __call__(self, params):      
    batch_size = params['batch_size']    
    data_dir = params['data_dir']
    file_pattern = os.path.join(data_dir, '*.tfrecords')
    dataset = tf.data.Dataset.list_files(file_pattern, shuffle=True)    
    dataset = dataset.shuffle(buffer_size=200)
    dataset = dataset.repeat()
    def fetch_dataset(filename):        
        dataset = tf.data.TFRecordDataset(filename, buffer_size= 8 * 1024 * 1024)
        return dataset
    dataset = dataset.apply(tf.contrib.data.parallel_interleave(fetch_dataset, cycle_length=8, sloppy=True))   
    dataset = dataset.shuffle(buffer_size=10000)         
    dataset = dataset.prefetch(8)
    dataset = dataset.map(parser_2, num_parallel_calls=8)
    dataset = dataset.batch(batch_size, drop_remainder=True)        
    dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
    images = dataset.make_one_shot_iterator().get_next()
    images = tf.reshape(images, [batch_size, 256, 256, 3])    
    random_noise = tf.random_normal([batch_size, self.noise_dim])
    features = {
        'real_images': images,
        'random_noise': random_noise}

    return features, None