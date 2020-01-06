#!/usr/bin/env python
import os
import sys
import cv2
import numpy as np

class video_decode:
    def __init__(self, file_path=None):
        self.src_file = file_path
        self.arg_check()
        self.env_init() 
    
    def arg_check(self):
        if not os.path.isfile(self.src_file):
            raise Exception("File not exist!")
            
    def env_init(self):
        cap = cv2.VideoCapture(self.src_file) 
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

    # Extract frames from a video and save them as jpg files; Frame ids in id_list should start from 0.
    def save_listed_frames(self, id_list=None, out_dir=None, verbose=True):
        # argument check
        if id_list is None : raise Exception("Frame id list is null!")
        if len(id_list) < 1: raise Exception("Frame id list length < 1")
        if out_dir is None: out_dir = os.path.join(os.getcwd(), 'save_frames')
        if not os.path.exists(out_dir): os.makedirs(out_dir)
        
        # main work
        cap = cv2.VideoCapture(self.src_file)
        fid = 0
        idx = 0
        extracted_cnt = 0
        while (True):
            ret, frame = cap.read()
            if frame is None:
                break
            else:
                fid += 1
                
            if fid == id_list[idx]:
                outname = os.path.join(out_dir, 'frame_%d.jpg' % fid)
                cv2.imwrite(outname, frame)
                extracted_cnt += 1
                idx += 1
                if verbose: print("%d: frame %d saved" % (extracted_cnt, fid+1))
                if idx == len(id_list): break
        cap.release()
        if verbose: print("Total saved frame number: {:d}".format(extracted_cnt))
        
    # Extract frames from a video and save them as jpg files; frame id starts from 0 with solid increment.
    def save_frames(self, gap=None, out_dir=None, verbose=True):
        """
        out_dir:  In. output directory for jpg files to save.
        gap    :  In. extract factor. gap=10 means save the first for every 10 frames. default=fps.
        """
        if gap is None: gap = self.fps    
        if out_dir is None: out_dir = os.path.join(os.getcwd(), 'save_frames')
        if not os.path.exists(out_dir): os.makedirs(out_dir)
            
        cap = cv2.VideoCapture(self.src_file)
        fid = 0
        extracted_cnt = 0
        while (True):
            ret, frame = cap.read()
            if frame is None:
                break
            if fid % gap == 0:
                outname = os.path.join(out_dir, 'frame_%d.jpg' % fid)
                cv2.imwrite(outname, frame)
                extracted_cnt += 1
                #cv2.imshow('frame_%d' % fid, frame)
            fid += 1
        cap.release()
        if verbose: print("Total saved frame number: {:d}".format(extracted_cnt))
            