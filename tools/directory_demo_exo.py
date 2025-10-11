import os
import argparse
import os.path as osp
from glob import glob
from collections import defaultdict
import json
from pathlib import Path
import time
import datetime
import shutil

import cv2
import torch
import joblib
import numpy as np
from loguru import logger
from progress.bar import Bar

from configs.config import get_cfg_defaults
from lib.data._custom import CustomDataset
from lib.utils.imutils import avg_preds
from lib.utils.transforms import matrix_to_axis_angle
from lib.models import build_network, build_body_model
from lib.models.preproc.detector import DetectionModel
from lib.models.preproc.extractor import FeatureExtractor
from lib.models.smplify import TemporalSMPLify
import sys
try: 
    from lib.models.preproc.slam import SLAMModel
    _run_global = True
except: 
    logger.info('DPVO is not properly installed. Only estimate in local coordinates !')
    _run_global = False

def run(cfg,
        video,
        output_pth,
        network,
        calib=None,
        run_global=True,
        save_pkl=False,
        visualize=False):
    
    cap = cv2.VideoCapture(video)
    assert cap.isOpened(), f'Faild to load video file {video}'
    fps = cap.get(cv2.CAP_PROP_FPS)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    # Whether or not estimating motion in global coordinates
    run_global = run_global and _run_global
    
    # Preprocess
    with torch.no_grad():
        if not (osp.exists(osp.join(output_pth, 'tracking_results.pth')) and 
                osp.exists(osp.join(output_pth, 'slam_results.pth'))):
            
            detector = DetectionModel(cfg.DEVICE.lower())
            extractor = FeatureExtractor(cfg.DEVICE.lower(), cfg.FLIP_EVAL)
            
            if run_global: slam = SLAMModel(video, output_pth, width, height, calib)
            else: slam = None
            
            bar = Bar('Preprocess: 2D detection and SLAM', fill='#', max=length)
            trk_ind = 0
            while (cap.isOpened()):
                flag, img = cap.read()
                
                if not flag: break
                
                # 2D detection and tracking
                detector.track(img, fps, length)

                # SLAM
                #"""
                if slam is not None: 
                    slam.track()
                #"""

                bar.next()
                trk_ind = trk_ind + 1
                if trk_ind > 180000:
                    break

            #logger.info(f"Image shape: {n_w} x {n_l}")

            tracking_results = detector.process(fps)
            # Get the number of boxes
            num_boxes = len(tracking_results)
            logger.info(f'Number of boxes: {num_boxes}')
            if num_boxes == 0:
                logger.info('No boxes detected. Skip the video.')
                with open("results/soccer_no_boxes.txt", "a") as f:
                    f.write(f"{video}\n")
                return False
            elif num_boxes > 1:
                # Get the box id of the biggest accumulated box area
                max_area = 0
                max_area_id = 0
                min_dis = 5000
                mid_dis_id = 0
                max_appear = 0
                max_appear_id = 0

                area_dict = {}
                dis_dict = {}
                appear_dict = {}
                ids = list(tracking_results.keys())
                for _id in ids:
                
                    bbox = tracking_results[_id]['bbox']
                    #logger.info(f'Box {_id}:  {tracking_results[_id]}') #{bbox}
                    appear = len(bbox)
                    logger.info(f'Box {_id} appears {appear} times')
                    area = sum(bbox)[2]
                    #logger.info(f'sum(bbox)[0]/appear: {sum(bbox)[0]/appear}, sum(bbox)[1]/appear: {sum(bbox)[1]/appear}, width/2: {width/2}, height/2: {height/2}')
                    mid_dis = abs(sum(bbox)[0]/appear - width / 2) + abs(sum(bbox)[1]/appear - height/2)
                    logger.info(f'Mid distance: {mid_dis}')
                    #logger.info(f'One box: {bbox[0][:]}')
                    #mid_dis = (bbox[0][0] + bbox[0][2] / 2) - n_l / 2
                    #logger.info(f'area: {area}')
                    if area not in area_dict.keys():
                        area_dict[area]  = [_id]
                    else:
                        area_dict[area].append(_id)

                    if mid_dis not in dis_dict.keys():
                        dis_dict[mid_dis] = [_id]
                    else:
                        dis_dict[mid_dis].append(_id)
                    if appear not in appear_dict.keys():
                        appear_dict[appear] = [_id]
                    else:
                        appear_dict[appear].append(_id)
                    #dis_dict[mid_dis] = _id
                    #appear_dict[appear] = _id

                    if appear > max_appear:
                        max_appear = appear
                        max_appear_id = _id
                    if area > max_area:
                        max_area = area
                        max_id = _id
                    if mid_dis < min_dis:
                        min_dis = mid_dis
                        mid_id = _id
                # Get the sorted dictionaries acoording to their keys
                area_dict = dict(sorted(area_dict.items(), key=lambda item: item[0], reverse=True))
                dis_dict = dict(sorted(dis_dict.items(), key=lambda item: item[0]))
                appear_dict = dict(sorted(appear_dict.items(), key=lambda item: item[0], reverse=True))
                logger.info(f'Area dict: {area_dict}')
                logger.info(f'Dis dict: {dis_dict}')
                logger.info(f'Appear dict: {appear_dict}')
                # Save the biggest box & box in the middle
                logger.info(f'Box {area_dict[max_area]} is the biggest box with area {max_area}')
                logger.info(f'Box {dis_dict[min_dis]} is the box in the middle')
                logger.info(f'Box {appear_dict[max_appear]} appears in most frame {max_appear} times')
                findone = False
                for dis_id in dis_dict[min_dis]:
                    if dis_id in area_dict[max_area]:
                        tracking_results = {dis_id: tracking_results[dis_id]}
                        findone = True
                        break
                if not findone:
                    for ap_id in appear_dict[max_appear]:
                        if ap_id in dis_dict[min_dis] or ap_id in area_dict[max_area]:
                            tracking_results = {ap_id: tracking_results[ap_id]}
                            findone = True
                            break
                            #tracking_results = {max_appear_id: tracking_results[max_appear_id]}
                if not findone:
                    logger.info('Contridicted boxes detected. Skip the video.')
                    with open("results/soccer_no_boxes.txt", "a") as f:
                        f.write(f"{video}\n")
                    return False
                #import ipdb
                #ipdb.set_trace()
                
                
                #return False


                #logger.info('Multiple boxes detected. Skip the video.')
                # save the video name to a file
                #with open("results/multiple_boxes.txt", "a") as f:
                #    f.write(f"{video}\n")
                #return False

            # First get the multi-box results!!!!
            #return True
            
            if slam is not None: 
                slam_results = slam.process()
            else:
                slam_results = np.zeros((length, 7))
                slam_results[:, 3] = 1.0    # Unit quaternion
        
            # Extract image features
            # TODO: Merge this into the previous while loop with an online bbox smoothing.
            
            tracking_results = extractor.run(video, tracking_results)
            logger.info('Complete Data preprocessing!')
            
            # Save the processed data
            joblib.dump(tracking_results, osp.join(output_pth, 'tracking_results.pth'))
            joblib.dump(slam_results, osp.join(output_pth, 'slam_results.pth'))
            logger.info(f'Save processed data at {output_pth}')
        
        # If the processed data already exists, load the processed data
        else:
            tracking_results = joblib.load(osp.join(output_pth, 'tracking_results.pth'))
            slam_results = joblib.load(osp.join(output_pth, 'slam_results.pth'))
            logger.info(f'Already processed data exists at {output_pth} ! Load the data .')
    
    # Build dataset
    dataset = CustomDataset(cfg, tracking_results, slam_results, width, height, fps)
    
    # run WHAM
    results = defaultdict(dict)
    
    n_subjs = len(dataset)
    for subj in range(n_subjs):

        with torch.no_grad():
            if cfg.FLIP_EVAL:
                # Forward pass with flipped input
                flipped_batch = dataset.load_data(subj, True)
                _id, x, inits, features, mask, init_root, cam_angvel, frame_id, kwargs = flipped_batch
                flipped_pred = network(x, inits, features, mask=mask, init_root=init_root, cam_angvel=cam_angvel, return_y_up=True, **kwargs)
                
                # Forward pass with normal input
                batch = dataset.load_data(subj)
                _id, x, inits, features, mask, init_root, cam_angvel, frame_id, kwargs = batch
                pred = network(x, inits, features, mask=mask, init_root=init_root, cam_angvel=cam_angvel, return_y_up=True, **kwargs)
                
                # Merge two predictions
                flipped_pose, flipped_shape = flipped_pred['pose'].squeeze(0), flipped_pred['betas'].squeeze(0)
                pose, shape = pred['pose'].squeeze(0), pred['betas'].squeeze(0)
                flipped_pose, pose = flipped_pose.reshape(-1, 24, 6), pose.reshape(-1, 24, 6)
                avg_pose, avg_shape = avg_preds(pose, shape, flipped_pose, flipped_shape)
                avg_pose = avg_pose.reshape(-1, 144)
                avg_contact = (flipped_pred['contact'][..., [2, 3, 0, 1]] + pred['contact']) / 2
                
                # Refine trajectory with merged prediction
                network.pred_pose = avg_pose.view_as(network.pred_pose)
                network.pred_shape = avg_shape.view_as(network.pred_shape)
                network.pred_contact = avg_contact.view_as(network.pred_contact)
                output = network.forward_smpl(**kwargs)
                pred = network.refine_trajectory(output, cam_angvel, return_y_up=True)
            
            else:
                # data
                batch = dataset.load_data(subj)
                _id, x, inits, features, mask, init_root, cam_angvel, frame_id, kwargs = batch
                
                # inference
                pred = network(x, inits, features, mask=mask, init_root=init_root, cam_angvel=cam_angvel, return_y_up=True, **kwargs)
        
        # if False:
        if args.run_smplify:
            smplify = TemporalSMPLify(smpl, img_w=width, img_h=height, device=cfg.DEVICE)
            input_keypoints = dataset.tracking_results[_id]['keypoints']
            pred = smplify.fit(pred, input_keypoints, **kwargs)
            
            with torch.no_grad():
                network.pred_pose = pred['pose']
                network.pred_shape = pred['betas']
                network.pred_cam = pred['cam']
                output = network.forward_smpl(**kwargs)
                pred = network.refine_trajectory(output, cam_angvel, return_y_up=True)
        
        # ========= Store results ========= #
        pred_body_pose = matrix_to_axis_angle(pred['poses_body']).cpu().numpy().reshape(-1, 69)
        pred_root = matrix_to_axis_angle(pred['poses_root_cam']).cpu().numpy().reshape(-1, 3)
        pred_root_world = matrix_to_axis_angle(pred['poses_root_world']).cpu().numpy().reshape(-1, 3)
        pred_pose = np.concatenate((pred_root, pred_body_pose), axis=-1)
        pred_pose_world = np.concatenate((pred_root_world, pred_body_pose), axis=-1)
        pred_trans = (pred['trans_cam'] - network.output.offset).cpu().numpy()
        
        results[_id]['pose'] = pred_pose
        results[_id]['trans'] = pred_trans
        results[_id]['pose_world'] = pred_pose_world
        results[_id]['trans_world'] = pred['trans_world'].cpu().squeeze(0).numpy()
        results[_id]['betas'] = pred['betas'].cpu().squeeze(0).numpy()
        results[_id]['frame_ids'] = frame_id
        results[_id]["width"] = width
        results[_id]["height"] = height
        focal_length = (width ** 2 + height ** 2) ** 0.5
        results[_id]["focal_length"] = focal_length
        if visualize:
            results[_id]['verts'] = (pred['verts_cam'] + pred['trans_cam'].unsqueeze(1)).cpu().numpy()
    
    if save_pkl:
        joblib.dump(results, osp.join(output_pth, "wham_output.pkl"))

    # Visualize
    if visualize:
        from lib.vis.run_vis import run_vis_on_demo
        with torch.no_grad():
            run_vis_on_demo(cfg, video, results, output_pth, network.smpl, vis_global=run_global)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--video', type=str, 
                        default='examples/demo_video.mp4', 
                        help='input video path or youtube link')

    parser.add_argument('--output_pth', type=str, default='output/demo', 
                        help='output folder to write results')
    
    parser.add_argument('--calib', type=str, default=None, 
                        help='Camera calibration file path')

    parser.add_argument('--estimate_local_only', action='store_true',
                        help='Only estimate motion in camera coordinate if True')
    
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize the output mesh if True')
    
    parser.add_argument('--save_pkl', action='store_true',
                        help='Save output as pkl file')
    
    parser.add_argument('--run_smplify', action='store_true',
                        help='Run Temporal SMPLify for post processing')

    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file('configs/yamls/demo.yaml')
    
    logger.info(f'GPU name -> {torch.cuda.get_device_name()}')
    logger.info(f'GPU feat -> {torch.cuda.get_device_properties("cuda")}')    
    
    # ========= Load WHAM ========= #
    smpl_batch_size = cfg.TRAIN.BATCH_SIZE * cfg.DATASET.SEQLEN
    smpl = build_body_model(cfg.DEVICE, smpl_batch_size)
    network = build_network(cfg, smpl)
    network.eval()
    
    # Output folder
    #pathlist = []
    #pathlist = Path(args.video).rglob("cam*.mp4")
    #dict_train_dance_path = './dict_train_dance_v0.json'
    #with open(dict_train_dance_path, 'r') as file:
    #    dict_train_dance_data = json.load(file)
    """
    for folder in dict_train_dance_data.keys():

        dir_path = os.path.join(args.video, folder)
        #print(f"dir_path: {dir_path}")
        pathlist0 = Path(dir_path).rglob("cam*.mp4")
        for path in pathlist0:
            path = str(path)
            #print(f"Process {path}")
            pathlist.append(path)
    """

    #sys.exit(0)

    #pathlist = Path(args.video).rglob("cam*.mp4")
    #only want the dance videos

    #print("Done globe")
    #pathlist = [path  for path in pathlist if "dance" in str(path)]
    #num = sum(1 for _ in pathlist)
    #print(f"Process {num} sequences!")
    # construct another generator as python only allow one pass
    #pathlist = Path(args.video).rglob("cam*.mp4")

    start_time = time.time()
    processed_num = 0
    total_num = 0
    num = 800
    #pathlist = ["../../egoexo/takes/uniandes_dance_002_10/frame_aligned_videos/cam01.mp4"]
    #for path in pathlist:
    print("Start processing.")
    for path in os.listdir("../../../../egoexo/takes"):
        #if "iiith_soccer" not in str(path):
        if "cmu_soccer" not in str(path):
            continue
        root_dir = os.path.join("takes", path)
        vid_path = "../../../../egoexo/" + root_dir + "/frame_aligned_videos"
        in_video = os.path.join("egoexo/train/soccer", path) 

        for ind in ["01", "02", "03", "04"]:
            input_video = os.path.join(in_video, ind)
            video_path = os.path.join(vid_path, f"cam{ind}.mp4")
            print("Processing ", video_path)
            save_dir = f"results/{input_video}"
            processed_wham = os.path.join(save_dir,  "wham_output.pkl") 
            #processed_wham = os.path.join(save_dir,  "output.mp4") 
            if os.path.exists(processed_wham):
                print (f"{processed_wham} already processed. Skip.")
                continue
            # multiple boxes detected, path name is written in multiple_boxes.txt
            # UPDATED: Process the biggest one
            #"""
            no_boxes = "results/soccer_no_boxes.txt"
            nob = False
            if os.path.exists(no_boxes):
                with open(no_boxes, "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        if input_video in line:
                            print(f"{input_video} has contridicted boxes detected. Skip.")
                            nob = True
            if nob:
                continue
            #"""
            
            print (save_dir)
            total_num += 1
            #continue

            os.makedirs(save_dir, exist_ok=True)

            #is_visualize = args.visualize and (processed_num % 100 == 0)
            is_visualize = args.visualize and (processed_num % 11 == 0)

            # ========= NOT Using DPVO WHAM ========= #
            path = video_path
            # inference
            can_infer = run(cfg, 
                path, 
                save_dir, 
                network, 
                args.calib, 
                #run_global=not args.estimate_local_only, 
                run_global=False,
                save_pkl=args.save_pkl,
                visualize=is_visualize)
            
            if can_infer == False:
                continue

            #print('\n Back to main script !\n')
            now_time = time.time()
            processed_num += 1
            used_time = now_time - start_time
            eta_time = 1.0 * used_time / processed_num * (num - processed_num)
            used_time = str(datetime.timedelta(seconds=int(used_time)))
            eta_time = str(datetime.timedelta(seconds=int(eta_time)))
            print(f"{processed_num} / {num}; used time: {used_time}, eta time: {eta_time} !")
        
        #break # for testing multi-boxes

    print(f"Total {total_num} sequences processed!")
    logger.info('Done !')