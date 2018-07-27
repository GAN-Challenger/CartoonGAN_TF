# -*- coding: utf-8 -*-


import tensorflow as tf
import tensorlayer as tl
import logging
import os
from util import get_imgs_fn, crop_sub_imgs_fn
import datetime
from model import generator, discriminator, vgg19_simple_api
import numpy as np
import math
import time
from random import shuffle

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')
logger = logging.getLogger(__name__)

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('real_img_path', '/data/real', """Directory to save realistic style image""")
tf.flags.DEFINE_string('cartoon_img_path', '/data/cartoon', """Directory to save cartoon style image""")
tf.flags.DEFINE_string('edge_img_path', '/data/edge', """Directory to save edge style image""")
tf.flags.DEFINE_string('vgg_model_path', '/home/liuzhaoyang/workspace/SRGAN_Wasserstein/vgg19.npy', """Path to save the VGG 19 parameters""")
tf.flags.DEFINE_boolean('edge_promote', True, """Integrate the edge promoting loss or not""")
tf.flags.DEFINE_float('loss_trade_off', 10.0, """Trade off ratio between adversarial loss and content loss""")
tf.flags.DEFINE_string('gpu', '0', """GPU device""")
tf.flags.DEFINE_string('mode', 'train', """Running mode, train | eveluate""")

tf.flags.DEFINE_integer('batch_size', 16,
                        """Number of batches to run.""")
tf.flags.DEFINE_float('lr_init', 1e-4,
                      """Init learning rate""")
tf.flags.DEFINE_integer('n_epoch_init', 10, """Pre-train iteration epochs""")
tf.flags.DEFINE_integer('n_epoch', 300, """Iteration epochs""")
tf.flags.DEFINE_integer('decay_every', 100, """Learning rate decay every %n epoch""")
tf.flags.DEFINE_float('lr_decay', 0.1, """Learning rate decay rate""")
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu


def read_all_imgs(img_list, path='', n_threads=32):
    """ Returns all images in array by given path and name of each image file. """
    imgs = []
    for idx in range(0, len(img_list), n_threads):
        b_imgs_list = img_list[idx: idx + n_threads]
        b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn, path=path)
        imgs.extend(b_imgs)
        logger.info('read %d from %s' % (len(imgs), path))
    count = FLAGS.batch_size * (len(imgs) // FLAGS.batch_size)
    return imgs[:count]


def main(argv):
    # init save directory
    save_dir_ginit = 'samples/{}_ginit'.format(FLAGS.mode)
    save_dir_gan = 'samples/{}_gan'.format(FLAGS.mode)
    log_dir = os.path.join('log', datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    tl.files.exists_or_mkdir(save_dir_ginit)
    tl.files.exists_or_mkdir(save_dir_gan)
    tl.files.exists_or_mkdir(log_dir)

    checkpoint_dir = 'checkpoint'
    tl.files.exists_or_mkdir(checkpoint_dir)

    # load data
    train_real_img_list = sorted(tl.files.load_file_list(path=FLAGS.real_img_path, regx='.*.png', printable=False))
    train_cartoon_img_list = sorted(
        tl.files.load_file_list(path=FLAGS.cartoon_img_path, regx='.*.png', printable=False))
    train_edge_img_list = sorted(tl.files.load_file_list(path=FLAGS.edge_img_path, regx='.*.png', printable=False))

    train_real_imgs = read_all_imgs(train_real_img_list, path=FLAGS.real_img_path, n_threads=32)
    train_cartoon_imgs = read_all_imgs(train_cartoon_img_list, path=FLAGS.cartoon_img_path, n_threads=32)
    train_edge_imgs = read_all_imgs(train_edge_img_list, path=FLAGS.edge_img_path, n_threads=32)

    logger.info('Load train real images size: %s' % len(train_real_imgs))   
 
    # define model
    img_real_input = tf.placeholder('float32', shape=[FLAGS.batch_size, 256, 256, 3], name='img_real_input')
    img_cartoon_input = tf.placeholder('float32', shape=[FLAGS.batch_size, 256, 256, 3], name='img_cartoon_input')
    img_edge_input = tf.placeholder('float32', shape=[FLAGS.batch_size, 256, 256, 3], name='img_edge_input')

    net_g = generator(img_real_input, is_train=True, reuse=False)
    net_d, img_true = discriminator(img_cartoon_input, is_train=True, reuse=False)
    _, img_gen_fake = discriminator(net_g.outputs, is_train=True, reuse=True)
    _, img_edge_fake = discriminator(img_edge_input, is_train=True, reuse=True)

    net_g.print_params(details=False)
    net_d.print_params(details=False)

    # vgg perceptual loss by conv4_4
    vgg_real = tf.image.resize_images(img_real_input, size=[224, 224], method=0, align_corners=False)
    vgg_gen = tf.image.resize_images(net_g.outputs, size=[224, 224], method=0, align_corners=False)

    net_vgg, vgg_real_emb = vgg19_simple_api((vgg_real + 1) / 2, reuse=False)
    _, vgg_gen_emb = vgg19_simple_api((vgg_gen + 1) / 2, reuse=True)

    # test inference
    net_g_test = generator(img_real_input, is_train=False, reuse=True)

    # loss definition
    with tf.name_scope('loss'):
        edge_promote = int(FLAGS.edge_promote)
        w = FLAGS.loss_trade_off

        adv_loss = tf.reduce_mean(tf.log(img_true)) + tf.reduce_mean(
            tf.log(1. - img_gen_fake)) + edge_promote * tf.reduce_mean(tf.log(1. - img_edge_fake))
        tf.summary.scalar('adv_loss', adv_loss)

        # L1 content loss
        con_loss = tf.reduce_mean(tf.abs(vgg_real - vgg_gen))
        tf.summary.scalar('con_loss', con_loss)

        d_loss = -1.0 * (adv_loss + w * con_loss)
        tf.summary.scalar('d_loss', d_loss)

        g_adv_loss = tf.reduce_mean(-1.0 * tf.log(img_gen_fake))
        tf.summary.scalar('g_adv_loss', g_adv_loss)

        g_loss = g_adv_loss + w * con_loss
        tf.summary.scalar('g_loss', g_loss)
    merged = tf.summary.merge_all()

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(FLAGS.lr_init, trainable=False)

    # optimizer
    g_vars = tl.layers.get_variables_with_name('CartoonGAN_G', True, True)
    d_vars = tl.layers.get_variables_with_name('CartoonGAN_D', True, True)
    g_opt_init = tf.train.RMSPropOptimizer(lr_v).minimize(con_loss, var_list=g_vars)
    g_opt = tf.train.RMSPropOptimizer(lr_v).minimize(g_loss, var_list=g_vars)
    d_opt = tf.train.RMSPropOptimizer(lr_v).minimize(d_loss, var_list=d_vars)

    # ##============================= Restore Model ===============================## #
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True
    sess = tf.Session(config=run_config)
    loss_writer = tf.summary.FileWriter(logdir=log_dir, graph=sess.graph)
    sess.run(tf.global_variables_initializer())

    if not tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g_{}.npz'.format(FLAGS.mode),
                                        network=net_g):
        tl.files.load_and_assign_npz(sess=sess,
                                     name=checkpoint_dir + '/g_{}_init.npz'.format(FLAGS.mode),
                                     network=net_g)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/d_{}.npz'.format(FLAGS.mode),
                                 network=net_d)

    # ##============================= Load VGG19 ===============================## #
    vgg19_npy_path = FLAGS.vgg_model_path
    if not os.path.isfile(vgg19_npy_path):
        logger.info("Please download vgg19.npz from : https://github.com/machrisaa/tensorflow-vgg")
        exit()
    npz = np.load(vgg19_npy_path, encoding='latin1').item()
    logger.info('** Success load VGG 19 network parameters **')

    params = []
    for val in sorted(npz.items()):
        W = np.asarray(val[1][0])
        b = np.asarray(val[1][1])
        logger.info("  Loading %s: %s, %s" % (val[0], W.shape, b.shape))
        params.extend([W, b])
    tl.files.assign_params(sess, params, net_vgg)

    # save sample images
    sample_imgs = train_real_imgs[:FLAGS.batch_size].copy()
    ni = int(math.sqrt(FLAGS.batch_size))
    sample_real_imgs = tl.prepro.threading_data(sample_imgs, fn=crop_sub_imgs_fn, is_random=False)
    logger.info('sample realistic sub-image: %s, %s, %s' % (sample_real_imgs.shape[0], sample_real_imgs.min(), sample_real_imgs.max()))
    tl.vis.save_images(sample_real_imgs, [ni, ni], save_dir_ginit + '/train_real.png')
    tl.vis.save_images(sample_real_imgs, [ni, ni], save_dir_gan + '/train_real.png')

    # ##============================= Pre Training ===============================## #
    sess.run(tf.assign(lr_v, FLAGS.lr_init))
    logger.info(" ** fixed learning rate: %f (for init G)" % FLAGS.lr_init)
    for epoch in range(FLAGS.n_epoch_init):
        epoch_time = time.time()
        total_content_loss, n_iter = 0., 0.

        for idx in range(0, len(train_real_imgs), FLAGS.batch_size):
            step_time = time.time()
            batch_real_imgs = tl.prepro.threading_data(
                train_real_imgs[idx: idx + FLAGS.batch_size],
                fn=crop_sub_imgs_fn, is_random=True
            )

            # update G
            content_error, _ = sess.run([con_loss, g_opt_init], {img_real_input: batch_real_imgs})
            logger.info("Epoch [%2d/%2d] %4d time: %4.4fs, content loss: %.8f " % (
                epoch, FLAGS.n_epoch_init, n_iter, time.time() - step_time, content_error))
            total_content_loss += content_error
            n_iter += 1
        logger.info("[*] Epoch: [%2d/%2d] time: %4.4fs, content loss: %.8f" % (
            epoch, FLAGS.n_epoch_init, time.time() - epoch_time, total_content_loss / n_iter))

        # quick evaluation on train set
        if epoch % 2 == 0:
            out = sess.run(net_g_test.outputs,
                           {img_real_input: sample_real_imgs})
            logger.info("[*] save images")
            tl.vis.save_images(out, [ni, ni], save_dir_ginit + '/train_%d.png' % epoch)

    # save pre train model
    tl.files.save_npz(net_g.all_params, name=checkpoint_dir + '/g_{}_init.npz'.format(FLAGS.mode),
                      sess=sess)

    # ##============================= GAN Training ===============================## #
    global_step = 0
    for epoch in range(FLAGS.n_epoch):
        if epoch != 0 and (epoch % FLAGS.decay_every == 0):
            new_lr_decay = FLAGS.lr_decay ** (epoch // FLAGS.decay_every)
            sess.run(tf.assign(lr_v, FLAGS.lr_init * new_lr_decay))
            logger.info(" ** new learning rate: %f (for GAN)" % (FLAGS.lr_init * new_lr_decay))
        elif epoch == 0:
            sess.run(tf.assign(lr_v, FLAGS.lr_init))
            log = " ** init lr: %f  decay_every_init: %d, lr_decay: %f (for GAN)" % (
                FLAGS.lr_init, FLAGS.decay_every, FLAGS.lr_decay)
            logger.info(log)

        epoch_time = time.time()
        total_d_loss, total_g_loss, n_iter = 0., 0., 0.

        shuffle(train_real_imgs)
        shuffle(train_cartoon_imgs)
        shuffle(train_edge_imgs)

        for idx in range(0, len(train_real_imgs), FLAGS.batch_size):
            step_time = time.time()
            batch_real_imgs = tl.prepro.threading_data(
                train_real_imgs[idx: idx + FLAGS.batch_size],
                fn=crop_sub_imgs_fn, is_random=True)
            batch_cartoon_imgs = tl.prepro.threading_data(
                train_cartoon_imgs[idx: idx + FLAGS.batch_size],
                fn=crop_sub_imgs_fn, is_random=True)
            batch_edge_imgs = tl.prepro.threading_data(
                train_edge_imgs[idx: idx + FLAGS.batch_size],
                fn=crop_sub_imgs_fn, is_random=True)

            # update D
            err_d, summary, _ = sess.run(
                [d_loss, merged, d_opt], feed_dict={
                    img_real_input: batch_real_imgs, img_cartoon_input: batch_cartoon_imgs,
                    img_edge_input: batch_edge_imgs
                }
            )
            loss_writer.add_summary(summary, global_step=global_step)
            # update G
            err_g, err_g_adv, err_con, _ = sess.run([g_loss, g_adv_loss, con_loss, g_opt],
                                                    {img_real_input: batch_real_imgs})

            logger.info("Epoch [%2d/%2d] %4d time: %4.4fs, d_loss: %.8f g_loss: %.8f (g_adv_loss: %.6f con_loss: %.6f)"
                        % (epoch, FLAGS.n_epoch, n_iter, time.time() - step_time, err_d, err_g, err_g_adv, err_con))
            total_d_loss += err_d
            total_g_loss += err_g
            n_iter += 1
            global_step += 1

        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, d_loss: %.8f g_loss: %.8f" % (
            epoch, FLAGS.n_epoch, time.time() - epoch_time, total_d_loss / n_iter, total_g_loss / n_iter)
        logger.info(log)

        # quick evaluation on train set
        if (epoch != 0) and (epoch % 10 == 0):
            out = sess.run(net_g_test.outputs,
                           {img_real_input: sample_real_imgs})
            logger.info("[*] save images")
            tl.vis.save_images(out, [ni, ni], save_dir_gan + '/train_%d.png' % epoch)

        # save model
        if (epoch != 0) and (epoch % 10 == 0):
            tl.files.save_npz(net_g.all_params, name=checkpoint_dir + '/g_{}.npz'.format(FLAGS.mode),
                              sess=sess)
            tl.files.save_npz(net_d.all_params, name=checkpoint_dir + '/d_{}.npz'.format(FLAGS.mode),
                              sess=sess)

if __name__ == '__main__':
    tf.app.run()
