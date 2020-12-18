from __future__ import division

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
from shutil import copyfile

# tf
import numpy as np
import tensorflow as tf
import torch

# save result
import face_alignment
import cv2
import PIL.Image as pil
import matplotlib.pyplot as plt
import trimesh

# path
_curr_path = os.path.abspath(__file__) # /home/..../face
_cur_dir = os.path.dirname(_curr_path) # ./

# save result
from src_common.common.face_io import write_self_camera, write_self_lm
from tools.preprocess.detect_landmark import LM_detector_howfar
from tools.preprocess.crop_image_affine import *

# graph
from src_tfGraph.build_graph import MGC_TRAIN

flags = tf.app.flags

#
flags.DEFINE_string("input_video", "data/test/my_test/", "Dataset directory")
flags.DEFINE_string("output_dir", "data/output_mytest", "Output directory")
flags.DEFINE_string("ckpt_file", "model/model-400000", "checkpoint file")
#flags.DEFINE_string("ckpt_file", "/home/jiaxiangshang/Downloads/202008/70_31_warpdepthepi_reg/model-400000", "checkpoint file")

#
flags.DEFINE_integer("batch_size", 1, "The size of of a sample batch")
flags.DEFINE_integer("img_width", 224, "Image(square) size")
flags.DEFINE_integer("img_height", 224, "Image(square) size")

# gpmm
flags.DEFINE_string("path_gpmm", "model/bfm09_trim_exp_uv_presplit.h5", "Dataset directory")
flags.DEFINE_integer("light_rank", 27, "3DMM coeffient rank")
flags.DEFINE_integer("gpmm_rank", 80, "3DMM coeffient rank")
flags.DEFINE_integer("gpmm_exp_rank", 64, "3DMM coeffient rank")

#
flags.DEFINE_boolean("flag_eval", True, "3DMM coeffient rank")
flags.DEFINE_boolean("flag_visual", True, "")
flags.DEFINE_boolean("flag_fore", False, "")

# visual
flags.DEFINE_boolean("flag_overlay_save", True, "")
flags.DEFINE_boolean("flag_overlayOrigin_save", True, "")
flags.DEFINE_boolean("flag_main_save", True, "")

FLAGS = flags.FLAGS

def main():
    FLAGS.input_video = os.path.join(_cur_dir, FLAGS.input_video)
    FLAGS.output_dir = os.path.join(_cur_dir, FLAGS.output_dir)

    FLAGS.ckpt_file = os.path.join(_cur_dir, FLAGS.ckpt_file)
    FLAGS.path_gpmm = os.path.join(_cur_dir, FLAGS.path_gpmm)
    
    
    if not os.path.exists(FLAGS.input_video):
        print("Error: no dataset_dir found")

    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    print("Finish copy")

    """
    preprocess
    """
    lm_d_hf = LM_detector_howfar(lm_type=int(3), device='cpu', face_detector='sfd')

    """
    build graph
    """
    import time
    time_st = time.time()
    system = MGC_TRAIN(FLAGS)
    system.build_test_graph(
        FLAGS, img_height=FLAGS.img_height, img_width=FLAGS.img_width, batch_size=FLAGS.batch_size
    )
    time_end = time.time()
    print("Time build: ", time_end - time_st)

    """
    load model
    """
    test_var = tf.global_variables()#tf.model_variables()
    # this because we need using the
    test_var = [tv for tv in test_var if tv.op.name.find('VertexNormalsPreSplit') == -1]
    saver = tf.train.Saver([var for var in test_var])

    #config = tf.ConfigProto()
    #Edited: tell tensorflow that we have 4 gpu available
    config=tf.ConfigProto(device_count = {'GPU': 4})
    config.gpu_options.allow_growth = True
    #Edited: limit the gpu memory usage
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    #Edited: allow tensorflow to choose the devices
    config.allow_soft_placement = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.graph.finalize()

        saver.restore(sess, FLAGS.ckpt_file)
        #
        path_video = FLAGS.input_video
        # Opens the Video file
        capVid = cv2.VideoCapture(path_video)
        print("VID OPENED")
        capCam = cv2.VideoCapture(0)
        print("CAM OPENED")

        frames=0
        while(capVid.isOpened() and capCam.isOpened()):
            frames += 1

            ret, vid_frame = capVid.read()
            if ret == False:
                print("Error: video failed ")
                break
            
            # Capture frame-by-frame
            ret, cam_frame = capCam.read()
            if ret == False:
                print("Error: camera failed ")
                break
            

            suceess = True
            # preprocess
            #image_rgb = vid_frame[..., ::-1]
            with torch.no_grad():
                lm_howfar_cam = lm_d_hf.lm_detection_howfar(cam_frame)
                try:
                    lm_howfar_cam = lm_howfar_cam[:, :2]
                except:
                    print("Error: detection camera failed ")
                    suceess = False
            
            exp_test = None
            if suceess:
                # face image align by landmark
                # we also provide a tools to generate 'std_224_bfm09'
                lm_trans, img_warped, tform = crop_align_affine_transform(lm_howfar, image_rgb, FLAGS.img_height, std_224_bfm09)
                image_rgb_b = img_warped[None, ...]
                """
                Start Camera inf
                """
                pred_cam = system.inference_exp_coeff(sess, image_rgb_b)
                exp_test = pred_cam['coeff_exp'][0]
                
            # preprocess
            #image_rgb = vid_frame[..., ::-1]
            with torch.no_grad():
                lm_howfar_vid = lm_d_hf.lm_detection_howfar(vid_frame)
                try:
                    lm_howfar_vid = lm_howfar_vid[:, :2]
                except:
                    print("Error: detection video failed ")
                    continue
            
            # face image align by landmark
            # we also provide a tools to generate 'std_224_bfm09'
            lm_trans, img_warped, tform = crop_align_affine_transform(lm_howfar, image_rgb, FLAGS.img_height, std_224_bfm09)
            image_rgb_b = img_warped[None, ...]
            # M_inv is used to back project the face reconstruction result to origin image
            M_inv = np.linalg.inv(tform.params)
            M = tform.params
            """
            Start Video inf
            """
            pred_vid = system.inference(sess, image_rgb_b, exp_coeff = exp_test)

            # name
            dic_image, name_image = os.path.split(path_image)
            name_image_pure, _ = os.path.splitext(name_image)

            """
            Render
            """
            image_input = image_rgb_b

            """
            NP
            """
            b = 0
            vertex_shape = pred_vid['vertex_shape'][0][b, :, :]
            vertex_color = pred_vid['vertex_color'][0][b, :, :]
            vertex_color = np.clip(vertex_color, 0, 1)
            #vertex_color_rgba = np.concatenate([vertex_color, np.ones([vertex_color.shape[0], 1])], axis=1)
            vertex_color_ori = pred_vid['vertex_color_ori'][0][b, :, :]
            vertex_color_ori = np.clip(vertex_color_ori, 0, 1)

            output_folder = os.path.join(FLAGS.output_dir, name_image_pure)

            if not os.path.exists(output_folder):
                os.mkdir(output_folder)

            """
            Common visual
            """
            if FLAGS.flag_visual:
                # visual
                #result_apper_mulPose_255 = pred_vid['apper_mulPose_255'][0][b, :, :]

                result_overlay_255 = pred_vid['overlay_255'][0][b, :, :]
                result_overlayGeo_255 = pred_vid['overlayGeo_255'][0][b, :, :]

                # common

                # mul poses
                # visual_concat = np.concatenate([image_input[0], result_overlay_255, result_overlayGeo_255, result_apper_mulPose_255], axis=1)
                # path_image_save = os.path.join(output_folder, name_image_pure + "_mulPoses.jpg")
                # plt.imsave(path_image_save, visual_concat)

                gpmm_render_mask = pred_vid['gpmm_render_mask'][0][b, :, :]
                gpmm_render_mask = np.tile(gpmm_render_mask, reps=(1, 1, 3))

                gpmm_render_overlay_wo = inverse_affine_warp_overlay(
                    M_inv, vid_frame, result_overlay_255, gpmm_render_mask)

                gpmm_render_overlay_gary_wo = inverse_affine_warp_overlay(
                    M_inv, vid_frame, result_overlayGeo_255, gpmm_render_mask)

                visual_concat = np.concatenate([vid_frame, gpmm_render_overlay_wo, gpmm_render_overlay_gary_wo], axis=1)

                cv2.imshow("Video", visual_concat)
                cv2.imshow("Camera", cam_frame)

        capVid.release()
        capCam.release()

if __name__ == '__main__':
    main()

