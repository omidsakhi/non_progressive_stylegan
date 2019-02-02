from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, ops, utils, glob, shutil

# Standard Imports
from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import
import numpy as np
import tensorflow as tf
from PIL import Image
import dataset_pipeline
import models
from tensorflow.contrib.tpu.python.tpu import tpu_config  # pylint: disable=E0611
from tensorflow.contrib.tpu.python.tpu import tpu_estimator  # pylint: disable=E0611
from tensorflow.contrib.tpu.python.tpu import tpu_optimizer  # pylint: disable=E0611
from tensorflow.python.estimator import estimator  # pylint: disable=E0611

FLAGS = flags.FLAGS

global dataset
dataset = dataset_pipeline

USE_TPU = False

# Cloud TPU Cluster Resolvers
flags.DEFINE_string(
    'tpu', default='TPU_NAME',
    help='The Cloud TPU to use for training. This should be either the name '
    'used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.')
flags.DEFINE_string(
    'gcp_project', default=None,
    help='Project name for the Cloud TPU-enabled project. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')
flags.DEFINE_string(
    'tpu_zone', default=None,
    help='GCE zone where the Cloud TPU is located in. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')

flags.DEFINE_string('data_dir', 'gs://BUCKET_NAME' if USE_TPU else r'D:/datasets/tfr-celeba256/train',
                    'Bucket/Folder that contains the data tfrecord files')
flags.DEFINE_string(
    'model_dir', 'gs://BUCKET_NAME/output' if USE_TPU else './output', 'Output model directory')
flags.DEFINE_integer('noise_dim', 256,
                     'Number of dimensions for the noise vector')
flags.DEFINE_integer('batch_size', 16,
                     'Batch size for both generator and discriminator')
flags.DEFINE_integer('num_shards', 8, 'Number of TPU chips')
flags.DEFINE_integer('train_steps', 2000000, 'Number of training steps')
flags.DEFINE_integer('train_steps_per_eval', 2000,
                     'Steps per eval and image generation')
flags.DEFINE_integer('iterations_per_loop', 50,
                     'Steps per interior TPU loop. Should be less than'
                     ' --train_steps_per_eval')
flags.DEFINE_string('mode', 'pred','pred vs train')
flags.DEFINE_float('learning_rate', 0.001, 'LR for both D and G')
flags.DEFINE_boolean('use_tpu', True if USE_TPU else False,
                     'Use TPU for training')
flags.DEFINE_integer('num_eval_images', 100,
                     'Number of images for evaluation')


def model_fn(features, labels, mode, params):
    del labels

    global_step = tf.train.get_or_create_global_step()

    g_alpha = tf.minimum(tf.cast(global_step, tf.float32) / 1000.0, 1.0)
    d_alpha = tf.minimum(tf.cast(global_step, tf.float32) / 100.0, 1.0)
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        ###########
        # PREDICT #
        ###########
        random_noise = features['random_noise']
        z00, z01, z10, z11 = models.mapping(random_noise)
        generated_images = models.generator(z00, z01, z10, z11, is_training=False)        
        ema, _ = utils.track_params_averages("Generator")
        predictions = {
            'generated_images': generated_images
        }

        if FLAGS.use_tpu:
            return tpu_estimator.TPUEstimatorSpec(mode=mode, predictions=predictions, prediction_hooks=[utils.RestoreParametersAverageValues(ema)])            
        else:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, prediction_hooks=[utils.RestoreParametersAverageValues(ema)])            

    batch_size = params['batch_size']   # pylint: disable=unused-variable
    real_images = features['real_images']     
    random_noise = features['random_noise']
    z00, z01, z10, z11 = models.mapping(random_noise)
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    generated_images = models.generator(z00, z01, z10, z11, is_training=is_training)
    generated_images_shuffled = tf.random_shuffle(generated_images)

    _, params_averages_op = utils.track_params_averages("Generator")

    d_real = models.discriminator(real_images)
    d_fake = models.discriminator(generated_images)    
    
    d_real_avg = tf.reduce_mean(d_real)
    d_fake_avg = tf.reduce_mean(d_fake)
    
    rg_ssim = tf.reduce_mean(tf.image.ssim_multiscale(generated_images, real_images, 2.0))
    g_ssim = tf.reduce_mean(tf.image.ssim_multiscale(generated_images, generated_images_shuffled, 2.0))
    ssim = rg_ssim / (g_ssim + 10e-8)    

    with tf.name_scope('Penalties'):
        d_loss_real = tf.reduce_mean(tf.nn.relu(1.0 - (d_real - d_fake_avg)))
        d_loss_fake = tf.reduce_mean(tf.nn.relu(1.0 + (d_fake - d_real_avg)))
        d_loss = d_loss_real + d_loss_fake        
        g_loss_fake = tf.reduce_mean(tf.nn.relu(1.0 - (d_fake - d_real_avg)))
        g_loss_real = tf.reduce_mean(tf.nn.relu(1.0 + (d_real - d_fake_avg)))
        g_loss = g_loss_fake + g_loss_real

    g_loss = tf.reduce_mean(g_loss)
    d_loss = tf.reduce_mean(d_loss)

    if FLAGS.use_tpu == False:
        tf.summary.image('generated_images', generated_images)
        tf.summary.image('real_images', real_images)        

        with tf.variable_scope('Out'):            
            tf.summary.scalar('real', tf.reduce_mean(d_real))
            tf.summary.scalar('fake', tf.reduce_mean(d_fake))            
        with tf.variable_scope('Loss'):            
            tf.summary.scalar('g_loss_fake', tf.reduce_mean(g_loss_fake))
            tf.summary.scalar('g_loss_real', tf.reduce_mean(g_loss_real))
            tf.summary.scalar('d_loss_real', tf.reduce_mean(d_loss_real))
            tf.summary.scalar('d_loss_fake', tf.reduce_mean(d_loss_fake))
        with tf.variable_scope('Acc'):            
            tf.summary.scalar('r_g_ssim', rg_ssim)
            tf.summary.scalar('g_g_ssim', g_ssim)
            tf.summary.scalar('ssim', ssim)            

    if mode == tf.estimator.ModeKeys.TRAIN:
        #########
        # TRAIN #
        #########
        d_optimizer = tf.train.AdamOptimizer(
            learning_rate=FLAGS.learning_rate * d_alpha, beta1=0.0, beta2=0.99)
        g_optimizer = tf.train.AdamOptimizer(
            learning_rate=FLAGS.learning_rate * g_alpha, beta1=0.0, beta2=0.99)

        if FLAGS.use_tpu:
            d_optimizer = tpu_optimizer.CrossShardOptimizer(d_optimizer)
            g_optimizer = tpu_optimizer.CrossShardOptimizer(g_optimizer)

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            d_step = d_optimizer.minimize(
                d_loss,
                var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                           scope='Discriminator'))
            g_step = g_optimizer.minimize(
                g_loss,
                var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                           scope='Generator'))

            with tf.control_dependencies([g_step]):
                params_averages_op = tf.group(params_averages_op)

            increment_step = tf.assign_add(
                tf.train.get_or_create_global_step(), 1)
            joint_op = tf.group([d_step, g_step, params_averages_op, increment_step])

            if FLAGS.use_tpu:
                return tpu_estimator.TPUEstimatorSpec(
                    mode=mode,
                    loss=g_loss,
                    train_op=joint_op)
            else:
                return tf.estimator.EstimatorSpec(
                    mode=mode,
                    loss=g_loss,
                    train_op=joint_op)

    elif mode == tf.estimator.ModeKeys.EVAL:
        pass

    raise ValueError('Invalid mode provided to model_fn')


def noise_input_fn(params, seed=None):
    noise_dataset = tf.data.Dataset.from_tensors(tf.constant(
        np.random.randn(params['batch_size'], FLAGS.noise_dim), dtype=tf.float32))
    noise = noise_dataset.make_one_shot_iterator().get_next()
    return {'random_noise': noise}, None

def noise_input_fn_fixed(params):    
    np.random.seed(0)
    noise_dataset = tf.data.Dataset.from_tensors(tf.constant(
        np.random.randn(params['batch_size'], FLAGS.noise_dim), dtype=tf.float32))
    noise = noise_dataset.make_one_shot_iterator().get_next()
    return {'random_noise': noise}, None

def main(argv):

    del argv

    tpu_cluster_resolver = None

    if FLAGS.use_tpu:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu,
            zone=FLAGS.tpu_zone,
            project=FLAGS.gcp_project)

        config = tpu_config.RunConfig(
            cluster=tpu_cluster_resolver,
            model_dir=FLAGS.model_dir,
            tpu_config=tpu_config.TPUConfig(
                num_shards=FLAGS.num_shards,
                iterations_per_loop=FLAGS.iterations_per_loop))

        est = tpu_estimator.TPUEstimator(
            model_fn=model_fn,
            use_tpu=FLAGS.use_tpu,
            config=config,
            params={"data_dir": FLAGS.data_dir},
            train_batch_size=FLAGS.batch_size,
            eval_batch_size=FLAGS.batch_size)

        local_est = tpu_estimator.TPUEstimator(
            model_fn=model_fn,
            use_tpu=False,
            config=config,
            params={"data_dir": FLAGS.data_dir},
            predict_batch_size=FLAGS.num_eval_images)
    else:
        est = tf.estimator.Estimator(
            model_fn=model_fn,
            model_dir=FLAGS.model_dir,
            params={"data_dir": FLAGS.data_dir, "batch_size": FLAGS.batch_size})

        local_est = tf.estimator.Estimator(
            model_fn=model_fn,
            model_dir=FLAGS.model_dir,
            params={"data_dir": FLAGS.data_dir, "batch_size": FLAGS.num_eval_images})

    tf.gfile.MakeDirs(os.path.join(FLAGS.model_dir))
    if FLAGS.mode == 'train':
        tf.gfile.MakeDirs(os.path.join(FLAGS.model_dir, 'generated_images'))
    else:
        tf.gfile.MakeDirs(os.path.join(FLAGS.model_dir, 'sampled_images'))
    tf.gfile.MakeDirs(os.path.join(FLAGS.model_dir, 'code'))        
    for _file in glob.glob(r'*.py'):        
        shutil.copy(_file, os.path.join(FLAGS.model_dir, 'code'))

    current_step = estimator._load_global_step_from_checkpoint_dir(
        FLAGS.model_dir)   # pylint: disable=protected-access,line-too-long
    if FLAGS.mode == 'train':
        tf.logging.info('Starting training for %d steps, current step: %d' %
                        (FLAGS.train_steps, current_step))
        while current_step < FLAGS.train_steps:
            next_checkpoint = min(current_step + FLAGS.train_steps_per_eval,
                                FLAGS.train_steps)
            est.train(input_fn=dataset.InputFunction(True, FLAGS.noise_dim),
                    max_steps=next_checkpoint)
            current_step = next_checkpoint
            tf.logging.info('Finished training step %d' % current_step)

            generated_iter = local_est.predict(input_fn=noise_input_fn_fixed)
            images = []
            for _ in range(FLAGS.num_eval_images):
                p = next(generated_iter)
                images.append(p['generated_images'][:, :, :])            
            filename = os.path.join(FLAGS.model_dir, 'generated_images', 'gen_%s.jpg' % (
                str(current_step).zfill(5)))
            utils.write_images(images, filename, [-1,1], 'RGB', 'JPEG')
            tf.logging.info('Finished generating images')
    elif FLAGS.mode == 'pred':
        count = 0        
        while count < 10:            
            generated_iter = local_est.predict(input_fn=noise_input_fn)
            images = []
            for _ in range(FLAGS.num_eval_images):
                p = next(generated_iter)
                images.append(p['generated_images'][:, :, :])            
            filename = os.path.join(FLAGS.model_dir, 'sampled_images', '%s_%s.jpg' % (str(current_step).zfill(5), str(count)))
            utils.write_images(images, filename, [-1,1], 'RGB', 'JPEG')
            count += 1            


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
