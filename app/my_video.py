#!/usr/bin/env python

# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A binary to train Inception on the flowers data set.
"""

from __future__ import absolute_import
from __future__ import division
#from __future__ import print_function

import os
import sys
import time
import glob
import video_op as vop
import file_op  as fop
from PIL import Image

def image_resize_aspect(pic_name, width=300, out_dir=None):
    """
    resize image with original aspect ratio.
    
    pic_name_list: In. picture name or name list.
    width        : expected width after resizing.
    """   
    
    if out_dir is None: out_dir = os.path.join(os.getcwd(), 'resized_image')
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    
    if isinstance(pic_name, list):
        for file in pic_name:
            img = Image.open(file)
            wpercent = (width / float(img.size[0]))
            hsize = int((float(img.size[1]) * float(wpercent)))
            img = img.resize((width, hsize), Image.ANTIALIAS)
            out = os.path.join(out_dir, file.split('/')[-1])
            img.save(out)
    else:
        img = Image.open(pic_name)
        wpercent = (width / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        img = img.resize((width, hsize), Image.ANTIALIAS)
        out = os.path.join(out_dir, pic_name.split('/')[-1])
        img.save(out)   


def extract_single_video():
    query = vop.video_decode('/home1/hvr2/crct_new/train_query/05d46d4a-b974-11e9-9c9e-fa163e3d9e3c.mp4')
    id_list = [1321,1377,1421,1468,1595,1630]
    query.save_listed_frames(id_list, "query")

    query = vop.video_decode('/home1/hvr2/crct_new/train_refer/2791972500.mp4')
    id_list = [123272,123366,123439,123517,123729,123788]
    query.save_listed_frames(id_list, "refer")

def extract_frames_for_train_refer(gap=10, file_list=None):
    src_root = '/home1/hvr2/crct_new/train_query'
    out_root = '/home1/hvr2/crct_new/train_query_image'
    flist = []
    cnt = 0
    
    t0 = time.clock()
    fop.recursive_file(src_root, flist)
    print "--- Extract frames for root: %s,  video num: %d ---" % (src_root, len(flist))
    for file in flist:
        cnt += 1
        if file_list is not None:
            if cnt < file_list[0] or cnt > file_list[1]: continue
        out_dir = os.path.join(out_root, file.split('/')[-1])
        refer = vop.video_decode(file)
        refer.save_frames(gap=gap, out_dir=out_dir, verbose=False)
        print "  %3d: %s is done." % (cnt, file)
    t1 = time.clock()
    print "Total time: %d sec" % (t1-t0)

def resize_images():
    src_root = '/home1/hvr2/crct_new/train_refer_image/1601734000.mp4'
    out_root = '/home1/hvr2/crct_new/train_refer_resized_image/1601734000.mp4'
    
    flist = glob.glob(src_root + '/*.jpg')
    image_resize_aspect(flist, out_dir=out_root)
        
    
if __name__ =='__main__':
    #refer = vop.video_decode('/home1/hvr2/crct_new/train_refer_crop/2817413400.mp4')
    #refer.save_frames(gap=10, out_dir='/home1/hvr2/crct_new/train_refer_crop_image/2817413400.mp4', verbose=False)

    #extract_single_video()
    #extract_frames_for_train_refer(gap=10, file_list=(2401,2999))
    #resize_images()
    

   
