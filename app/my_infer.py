#!/usr/bin/env python

# from __future__ import division
import sys
import os
import time
import datetime
import struct
import caffe
import scipy
import numpy as np
import video_op as vop
import file_op  as fop
import progress_op as pop
from numpy import random as nr
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize as sknormalize

my_root='/home1/landlee/data/crct'
caffe_root='/home1/landlee/BAK/my_lib/python27/caffe'

def save_vlad(vlad_file_name, vlad_data, dim_num):
    with open(vlad_file_name, 'wb') as fd:
        fd.write(struct.pack('i', 1))
        fd.write(struct.pack('i', dim_num))
        for i in range(len(vlad_data)):
            fd.write(struct.pack('f', vlad_data[i]))
        fd.close()       

def read_vald(vlad_file_name, dim_num):
    with open(vlad_file_name, 'rb') as fd:
        file = fd.read()
        data = struct.unpack('ii%df' % dim_num, file)
        array = np.array(data[2:-1])
        fd.close()
    return  array

class caffe_model():
    def __init__(self, model_id=0, model_init=1):
        self.model_id = model_id
        self.caffe_root = caffe_root
        mean_file = self.caffe_root + '/data/ilsvrc12/mean.npy'
        if model_id == 0:   # VGG16
            caffe_model = self.caffe_root +'/models/vgg16/VGG_ILSVRC_16_layers.caffemodel'
            deploy = self.caffe_root + '/models/vgg16/VGG_ILSVRC_16_layers_deploy_cut.prototxt'
        elif model_id ==1 : # HVR
            caffe_model = self.caffe_root +'/models/hvr2/liu_alexnet_train_iter_5000.caffemodel'
            deploy = self.caffe_root + '/models/hvr2/deploy.prototxt'
        elif model_id ==2 : # Alexnet
            caffe_model = self.caffe_root +'/models/bvlc_alexnet/bvlc_alexnet.caffemodel'
            deploy = self.caffe_root + '/models/bvlc_alexnet/deploy_cut.prototxt'
        else:
            raise Exception("Not supported model id (%d)" % model_id)
            
        self.gaussian_kernel_init()
        if model_init: self.model_init(caffe_model, deploy, mean_file)
        
    def model_init(self, caffe_model, deploy, mean_file):
        sys.path.insert(0, self.caffe_root + '/python')
        caffe.set_mode_gpu()
        caffe.set_device(0)
        net = caffe.Net(deploy, caffe_model, caffe.TEST)
        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2, 0, 1))
        transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))
        transformer.set_raw_scale('data', 255)
        transformer.set_channel_swap('data', (2, 1, 0))
        self.net = net
        self.transformer = transformer

    def pca_init(self):
        self.pca = PCA(n_components=128)
        self.pca.fit(X)
        
    def gaussian_kernel_init(self):
        if self.model_id == 0:
            self.generate_gaussian_weight_matrix(512,14,14)
        elif self.model_id == 1:
            self.kernel = None
        elif self.model_id == 2:
            self.generate_gaussian_weight_matrix(256,13,13)

    def dump_kernel(self, W, H):
        print "sigma = %.1f" % self.sigma
        for r in range(W):
            for c in range(H):
                print "%.6f " % self.kernel[0,r,c],
            print ""
        exit()
            
    def generate_gaussian_weight_matrix(self, C, W, H):
        self.sigma = 2.8
        c_x = (W+1)/2.0
        c_y = (H+1)/2.0
        den = 2*(self.sigma**2)
        
        # original Gaussian kernel
        kernel = np.empty([C, W, H], dtype=float)
        for x in range(1, W+1):
            for y in range(1, H+1):
                sum = (x-c_x)**2 + (y-c_y)**2
                kernel[:,x-1,y-1] = scipy.exp(-sum/den)

        '''
        # New Method                
        line=4
        sigma = 2.0;den = 2*(sigma**2)        
        for x in range(1, W+1):
            if x <=line or x>= W+1-line:
                for y in range(1, H+1):
                    sum = (x-c_x)**2 + (y-c_y)**2
                    kernel[:,x-1,y-1] = scipy.exp(-sum/den)
        for y in range(1, H+1):
            if y<=line or y>=H+1-line:        
                for x in range(1, W+1):
                    sum = (x-c_x)**2 + (y-c_y)**2
                    kernel[:,x-1,y-1] = scipy.exp(-sum/den)
        '''
        
        self.kernel = kernel
        #self.dump_kernel(W, H)

    def l2_dist(self, a, b):
        return np.linalg.norm(a-b)

    def get_cnn_feat(self, image_name):
        image = caffe.io.load_image(image_name)        
        self.net.blobs['data'].data[...] = self.transformer.preprocess('data', image)
        self.net.forward()
        if self.model_id == 0:
            #feat = self.net.blobs['pool5'].data[0]
            feat = self.net.blobs['conv5_3'].data[0];
        elif self.model_id == 1: # HVR
            feat = self.net.blobs['fc6'].data[0]
            feat = sknormalize(np.reshape(feat,(1,-1)))[0]
        elif self.model_id == 2: # Alexnet        
            feat = self.net.blobs['conv5'].data[0]            
        return feat        
        
    def compute_crow_spatial_weight(self, X, a=2, b=2):
        """
        Given a tensor of features, compute spatial weights as normalized total activation.
        Normalization parameters default to values determined experimentally to be most effective.

        :param ndarray X:
            3d tensor of activations with dimensions (channels, height, width)
        :param int a:
            the p-norm
        :param int b:
            power normalization
        :returns ndarray:
            a spatial weight matrix of size (height, width)
        """
        S = X.sum(axis=0)
        z = (S**a).sum()**(1./a)
        return (S / z)**(1./b) if b != 1 else (S / z)

    def compute_crow_channel_weight(self, X):
        """
        Given a tensor of features, compute channel weights as the
        log of inverse channel sparsity.

        :param ndarray X:
            3d tensor of activations with dimensions (channels, height, width)
        :returns ndarray:
            a channel weight vector
        """
        K, w, h = X.shape
        area = float(w * h)
        nonzeros = np.zeros(K, dtype=np.float32)
        for i, x in enumerate(X):
            nonzeros[i] = np.count_nonzero(x) / area

        nzsum = nonzeros.sum()
        for i, d in enumerate(nonzeros):
            nonzeros[i] = np.log(nzsum / d) if d > 0. else 0.

        return nonzeros
        
    def spoc_aggregation(self, X):
        """
        Given a tensor of activations, compute the aggregate Spoc feature, weighted
        spatially and channel-wise.

        :param ndarray X:
            3d tensor of activations with dimensions (channels, height, width)
        :returns ndarray:
            Spoc aggregated global image feature    
        """  
        # original method
        '''
        X = X * self.kernel
        S = X.sum(axis=(1,2))
        return sknormalize(S.reshape(1,-1))[0]
        '''
        
        # New method1: spoc + Ch_weight
        ker = self.kernel[:,1:13,1:13]
        X = X[:,1:13,1:13]
        C = self.compute_crow_channel_weight(X)
        X = X * ker
        X = X.sum(axis=(1,2))
        X = X * C
        '''

        # New method2: spoc + Ch_weight + rmac
        C = self.compute_crow_channel_weight(X)
        X = X * self.kernel                
        L0 = X.sum(axis=(1,2))
        L1 = X[:,].
        X = X * C
        '''
        
        return sknormalize(X.reshape(1,-1))[0]

    def crow_aggregation(self, X):
        """
        Given a tensor of activations, compute the aggregate Spoc feature, weighted
        spatially and channel-wise.

        :param ndarray X:
            3d tensor of activations with dimensions (channels, height, width)
        :returns ndarray:
            Spoc aggregated global image feature    
        """  
        S = self.compute_crow_spatial_weight(X)
        C = self.compute_crow_channel_weight(X)
        X = X * S
        X = X.sum(axis=(1, 2))
        X = X * C
        return sknormalize(np.reshape(X,(1,-1)))[0]      
       
    # infer one picture and save its eigenvector
    def single_infer(self, image_name, out_dir=None):
        if not fop.is_file(image_name): raise Exception("File(%s) not exist!" % image_name)
        if out_dir is None: out_dir = os.path.join(os.getcwd(), 'vgg16_feats')
        if not os.path.exists(out_dir): os.makedirs(out_dir)
        
        feat = self.get_cnn_feat(image_name)
        npy_fname = image_name.split('/')[-1].replace('jpg','npy')
        npy_file = os.path.join(out_dir, npy_fname)
        np.save(npy_file, feat)

    # infer one picture and save its crow eigenvector
    def single_infer_with_crow(self, image_name, out_dir=None):
        if not fop.is_file(image_name): raise Exception("File(%s) not exist!" % image_name)
        if out_dir is None: out_dir = os.path.join(os.getcwd(), 'crow_feats')
        if not os.path.exists(out_dir): os.makedirs(out_dir)
        image = caffe.io.load_image(image_name)
        
        self.net.blobs['data'].data[...] = self.transformer.preprocess('data', image)
        self.net.forward()
        pool5 = self.net.blobs['pool5'].data[0]; feat = self.crow_aggregation(pool5)
        npy_file = os.path.join(out_dir, image_name.split('/')[-1].replace('jpg','npy'));  np.save(npy_file, feat)
        return feat

    # infer one picture and save its spoc eigenvector
    def single_infer_with_spoc(self, image_name, out_dir=None):
        if not fop.is_file(image_name): raise Exception("File(%s) not exist!" % image_name)
        if out_dir is None: out_dir = os.path.join(os.getcwd(), 'spoc_feats')
        if not os.path.exists(out_dir): os.makedirs(out_dir)
        
        cov5 = self.get_cnn_feat(image_name)
        feat = self.spoc_aggregation(cov5)
        #npy_fname = image_name.split('/')[-1].replace('jpg','npy'); npy_file = os.path.join(out_dir, npy_fname); np.save(npy_file, feat)
        return feat

       
def test_pair_images():
    name1 = '1080p_1_0.jpg' #'/home1/hvr2/crct_new/train_refer_image/1601734000.mp4/frame_0.jpg'
    name2 = '1080p_1_convert_to_1066_600_0.jpg' #'/home1/hvr2/crct_new/train_refer_crop_image/1601734000.mp4/frame_10.jpg'
    out_dir = '.'
    model = caffe_model()
    f1 = model.single_infer_with_crow(name1, out_dir)
    f2 = model.single_infer_with_crow(name2, out_dir)
    dist = model.l2_dist(f1, f2)
    print 'dist: %f' % dist

def test_pair_videos():
    root1 = '/home1/hvr2/crct_new/train_refer_image/1601734000.mp4'
    #root2 = '/home1/hvr2/crct_new/train_refer_crop_image/1601734000.mp4'
    root2 = '/home1/hvr2/crct_new/train_refer_resized_image/1601734000.mp4'
    flist = os.listdir(root1)
    
    model = caffe_model()
    for idx in range(len(flist)):
        fname = "frame_%d.jpg" % (idx*10)
        f1 = model.single_infer_with_spoc(os.path.join(root1, fname))
        f2 = model.single_infer_with_spoc(os.path.join(root2, fname))
        print 'frame_%d.jpg: %f' % (idx*10, model.l2_dist(f1, f2))
    
def test_pair_feat():
    name1 = '/home1/hvr2/crct_new/train_refer_spoc_feat/1601734000.mp4/frame_0.npy'
    name2 = '/home1/hvr2/crct_new/train_refer_crop_spoc_feat/1601734000.mp4/frame_0.npy'

    model = caffe_model(model_init=0)
    f1 = np.load(name1)
    f2 = np.load(name2)
    dist = model.l2_dist(f1, f2)
    print 'dist: %f' % dist


def save_crow_feat(file_list=None):
    src_base = os.path.join(my_root, 'train_refer_feat_alex')
    out_base = os.path.join(my_root, 'train_refer_crop_feat_crow')
    dlist = []
    dir_cnt = 0
    
    t0 = time.time()
    fop.recursive_dir(src_base, dlist, depth=0)
    
    model = caffe_model(model_id=model_id, model_init=0)
    for dir in dlist:
        dir_cnt += 1
        if file_list is not None:
            if dir_cnt < file_list[0] or dir_cnt > file_list[1]: continue
        flist = os.listdir(dir)
        print("  --- Crow feat: %d: %s   frame: %d" % (dir_cnt,dir, len(flist)))
        out_dir = os.path.join(out_base, dir.split('/')[-1])
        if not os.path.exists(out_dir): os.makedirs(out_dir)
        for f in flist:
            src = os.path.join(dir, f)
            x = np.load(src)
            crow = model.crow_aggregation(x)
            dst = os.path.join(out_dir, src.split('/')[-1])
            np.save(dst, crow)
            #print("===SHAPE:",x.shape,crow.shape,crow);exit()
    t1 = time.time()
    print("Total time: %d sec" % (t1-t0))
    
    
def extract_conv_feat(file_list=None, model_id=0):
    model = caffe_model(model_id=model_id)
    if model_id == 0: suffix = 'vgg'
    elif model_id == 2: suffix = 'alex'
    for cnt in range(2):
        if cnt:
            src_base = os.path.join(my_root, 'train_refer_crop_image')
            out_base = os.path.join(my_root, 'train_refer_crop_%s' % suffix)
        else:
            src_base = os.path.join(my_root, 'train_refer_image')
            out_base = os.path.join(my_root, 'train_refer_%s' % suffix)

        dlist = []
        dir_cnt = 0
        if not os.path.exists(out_base): os.makedirs(out_base)
        
        t0 = time.time()
        fop.recursive_dir(src_base, dlist, depth=0)

        for dir in dlist:
            dir_cnt += 1
            if file_list is not None:
                if dir_cnt < file_list[0] or dir_cnt > file_list[1]: continue
            
            flist = os.listdir(dir)
            print("  --- Conv feat: %d: %s   frame: %d" % (dir_cnt,dir, len(flist)))
            out_dir = os.path.join(out_base, dir.split('/')[-1])
            for f in flist:
                file = os.path.join(dir, f)
                model.single_infer(file, out_dir)  
        t1 = time.time()
        print("src_base: %s" % src_base)
        print("out_base: %s" % out_base)
        print("Total time: %d sec" % (t1-t0))
        print("--- extract conv feat End ---")
    
def thread_spoc(tid, model_id, file_list, dlist, out_base):
    print "Task [%2d] running, dir range: %d ~ %d" %(tid, file_list[0], file_list[1])
    
    model = caffe_model(model_id=model_id, model_init=0)
    dir_cnt = 0
    for dir in dlist:
        dir_cnt += 1       
        if dir_cnt < file_list[0]: continue
        if dir_cnt > file_list[1]: break
        flist = os.listdir(dir)
        print("  --- [%d] Spoc feat: %d: %s   frame: %d" % (tid, dir_cnt, dir, len(flist)))
        out_dir = os.path.join(out_base, dir.split('/')[-1])
        if not os.path.exists(out_dir): os.makedirs(out_dir)
        for f in flist:
            src = os.path.join(dir, f)
            x = np.load(src)
            spoc = model.spoc_aggregation(x)
            dst = os.path.join(out_dir, src.split('/')[-1])
            np.save(dst, spoc) 
    
def extract_spoc_feat(file_list=None, model_id=0, thread_num=1):
    if model_id == 1: return
    prefix = ['train_refer_', 'train_refer_crop_']
    src_sufix  = ['vgg', '', 'alex']
    dst_sufix  = ['vggspoc', '', 'alexspoc']
    model = caffe_model(model_id=model_id, model_init=0)
    print "--- extract spoc feat Start ---"
    
    for cnt in range(2):      
        src_base = os.path.join(my_root, prefix[cnt] + src_sufix[model_id])
        out_base = os.path.join(my_root, prefix[cnt] + dst_sufix[model_id])
        dlist = []
        dir_cnt = 0
        if not os.path.exists(out_base): os.makedirs(out_base)
        
        t0 = time.time()
        fop.recursive_dir(src_base, dlist, depth=0)
        print "Dir number: ",len(dlist)
        
        task_list = []
        total = len(dlist)
        if total <= thread_num: thread_num = total
        each  = total / thread_num
        for tid in range(thread_num):
            start = tid*each + 1
            end   = start + each - 1
            if tid+1 == thread_num: end = total
            task = pop.task(func=thread_spoc, args=(tid+1, model_id, (start,end), dlist, out_base))
            task.run()
            task_list.append((tid+1, task))
        
        for tid, task in task_list:
            task.close()
            print "Task [%d] exited." % tid           
              
        t1 = time.time()
        print("src_base: %s" % src_base)
        print("out_base: %s" % out_base)
        print("model_id: %d,  sigma: %.1f" % (model_id, model.sigma))
        print("Total time: %d sec" % (t1-t0))
        
    print("--- extract spoc feat End ---")   

def feat_nonzero_stat():
    feat_num = 1000
    src_base = os.path.join(my_root, 'train_refer_feat_vggspoc')
    sample_flist = []
    fop.recursive_file(src_base, sample_flist) 
    for idx in range(len(sample_flist)):
        feat = np.load(sample_flist[idx])
        feat = feat > 0.
        print "%4d: %d" % (idx+1, feat.sum())
    
def save_train_feat():
    '''
    feat_num = 1000
    src_base = os.path.join(my_root, 'train_refer_feat_vggspoc')
    sample_flist = []
    fop.recursive_file(src_base, sample_flist)
    print "src_base: %s,  feature_num: %d" % (src_base, len(sample_flist))
    feat_list = np.load(sample_flist[1000])
    for idx in range(1000,len(sample_flist)):
        if idx >= 999+feat_num: break
        feat_list = np.append(feat_list, np.load(sample_flist[idx]))
            
    print "Feature loaded done. size=%d" % feat_list.shape
    feat_list = np.reshape(feat_list, (feat_num,512))  
    np.save(os.path.join(my_root, 'pca_train_%d.npy' % feat_num), feat_list)    
    print "Feature saved  done."
    '''
    feat_list = np.load(os.path.join(my_root, 'pca_train_%d.npy' % 1000))
    pca = PCA(n_components=256)
    pca.fit(feat_list)
    print pca.explained_variance_ratio_
    print pca.explained_variance_

if __name__ =='__main__':
    model_id = 1  # 0-VGG16, 1-HVR, 2-Alexnet
    #feat_nonzero_stat()
    #save_train_feat()
    #print "Program start: ", time.asctime(time.localtime())
    #test_pair_images()
    #test_pair_videos()
    #test_pair_feat()
    #extract_feat_from_image_base()
    #save_crow_feat(file_list=(1,5), model_id=model_id)
    #extract_conv_feat(file_list=(1,5), model_id=model_id)
    extract_spoc_feat(file_list=(1,5), model_id=model_id, thread_num=10)  
    import my_eval as eva;eva.evaluate_multi(thread_num=20, model_id=model_id, topN=5)
   
    
    
