import tensorflow as tf
import ops

def discriminator(x, is_training=True, scope='Discriminator'):
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):

    x = tf.concat(tf.split(x, 2, 0), axis=3)

    fn = 64    

    x = ops.conv2d_down('256-128', x, fn, 5, 'NHWC')
    x = ops.lrelu(x)

    x = ops.conv2d_down('128-64', x, fn * 2, 5, 'NHWC')
    x = ops.lrelu(x)

    x = ops.conv2d_down('64-32', x, fn * 4, 5, 'NHWC')
    x = ops.lrelu(x)

    x = ops.conv2d_down('32-16', x, fn * 4, 5, 'NHWC')
    x = ops.lrelu(x)

    x = ops.conv2d_down('16-8', x, fn * 4, 5, 'NHWC')
    x = ops.lrelu(x)
    
    x = ops.minibatch_stddev_layer(x, 'NHWC')

    x = ops.conv2d_down('8-4', x, fn * 8, 5, 'NHWC')
    x = ops.lrelu(x)

    x = ops.conv2d('4', x, 1, 1, 'NHWC', gain=1.0)

    return x

def gblock_top(z1, z2, ch_dim, name, lr_mult1=1.0, lr_mult2=1.0):
    with tf.variable_scope(name):
        batch_size = int(z1.get_shape()[0])
        top8 = tf.get_variable('top', shape=[1, 8, 8, ch_dim], dtype=tf.float32)
        top8 = ops._lr_mult(lr_mult2)(top8)
        x = tf.tile(top8, [batch_size, 1, 1, 1])
        x = ops.lrelu(ops.adaptive_instance_norm('top_bn', x, z1, 'NHWC', lr_mult=lr_mult2))        
        x = ops.conv2d("conv_2", x, ch_dim, 3, 'NHWC', has_bias=False, lr_mult=lr_mult1)
        x = ops.lrelu(ops.adaptive_instance_norm('bn_2', x, z2, 'NHWC', lr_mult=lr_mult1))        
        return x

def gblock_up(x, z1, z2, ch_dim, kernel_size, name, lr_mult1=1.0, lr_mult2=1.0):
    with tf.variable_scope(name):                
        x = ops.upscale2d(x, 'NHWC')
        x = ops.conv2d("conv_1", x, ch_dim, kernel_size, 'NHWC', has_bias=False, lr_mult=lr_mult2)
        x = ops.lrelu(ops.adaptive_instance_norm('bn_1', x, z1, 'NHWC', lr_mult=lr_mult2))        
        x = ops.conv2d("conv_2", x, ch_dim, kernel_size, 'NHWC', has_bias=False, lr_mult=lr_mult1)
        x = ops.lrelu(ops.adaptive_instance_norm('bn_2', x, z2, 'NHWC', lr_mult=lr_mult1))
        return x

def mapping(z, scope='Generator'):
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):        
  
        fn = 32
        steps = 500

        z = ops.dense('fc', z, fn * 4, 'NC', lr_mult=1.0/(16.0 * steps))        
        z = ops.relu(z)
        
        z0 = ops.dense('fc-0', z, fn * 4, 'NC', lr_mult=1.0/(15.0 * steps))        
        z0 = ops.relu(z0)

        z1 = ops.dense('fc-1', z, fn * 4, 'NC', lr_mult=1.0/(15.0 * steps))        
        z1 = ops.relu(z1)

        z00 = ops.dense('fc-00', z0, fn * 4, 'NC', lr_mult=1.0/(14.0 * steps))        
        z00 = ops.relu(z00)

        z01 = ops.dense('fc-01', z0, fn * 4, 'NC', lr_mult=1.0/(14.0 * steps))        
        z01 = ops.relu(z01)

        z10 = ops.dense('fc-10', z1, fn * 4, 'NC', lr_mult=1.0/(14.0 * steps))        
        z10 = ops.relu(z10)

        z11 = ops.dense('fc-11', z1, fn * 4, 'NC', lr_mult=1.0/(14.0 * steps))        
        z11 = ops.relu(z11)

  return z00, z01, z10, z11

def generator(z00, z01, z10, z11, is_training, scope='Generator'):  
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):        
        
        fn = 32
        steps = 500

        x = gblock_top(z00,z00, fn * 8, 'top', lr_mult1=1.0/(12.0 * steps), lr_mult2=1.0/(13.0 * steps))
        x = gblock_up(x, z00,z01, fn * 8, 3, '8-16', lr_mult1=1.0/(10.0 * steps), lr_mult2=1.0/(11.0 * steps))
        x = gblock_up(x, z01,z01, fn * 4, 3, '16-32', lr_mult1=1.0/(8.0 * steps), lr_mult2=1.0/(9.0 * steps))
        x = gblock_up(x, z10,z10, fn * 4, 3, '32-64', lr_mult1=1.0/(6.0 * steps), lr_mult2=1.0/(7.0 * steps))
        x = gblock_up(x, z10,z11, fn * 2, 3, '64-128', lr_mult1=1.0/(4.0 * steps), lr_mult2=1.0/(5.0 * steps))
        x = gblock_up(x, z11,z11, fn, 3, '128-256', lr_mult1=1.0/(2.0 * steps), lr_mult2=1.0/(3.0 * steps))
        x = ops.conv2d('gout', x, 3, 1, 'NHWC', gain=1.0, lr_mult=1.0/(1.0 * steps))

        x  = tf.nn.tanh(x)

    return x