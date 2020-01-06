#!/usr/bin/env python

import os
import sys
import glob
import time
import scipy
import numpy as np
import file_op  as fop
import progress_op as pop
#import threading


my_root='/home1/landlee/data/crct'

class eval_param():
    def __init__(self):
        self.query_flist = []
        self.sample_lib  = []
        self.valid = 0
        self.perfect = 0
        self.id_error = 0
        self.name_error = 0
        self.perfect_d  = 0.
        self.valid_d    = 0.
        self.name_error_d = 0.
        self.id_error_d   = 0.
        self.id_error_list = []
        self.name_error_list = []

def get_video_name(file):
    return file.split('/')[-2]

def get_frame_id(file):
    name = file.split('/')[-1]
    start = name.rindex('_') + 1
    end = name.rindex('.')
    return int(name[start:end])

def evaluate(sample, query, mode='spoc', tolerance=75):
    sample_lib = []
    perfect    = 0
    valid      = 0
    name_error  = 0
    id_error = 0
    name_error_list = []
    id_error_list = []
    
    flist = []
    frame_cnt = 0
    t0 = time.time()    
    fop.recursive_file(sample, flist)
    #print flist; print get_frame_id(flist[1]); exit()
    for file in flist:
        frame_cnt += 1
        sample_lib.append((np.load(file), file))
    print "Sample : %s, frame number: %d" % (sample, frame_cnt)
    
    flist = []
    frame_cnt = 0
    fop.recursive_file(query, flist)
    print "Query  : %s, frame number: %d" % (query, len(flist))
    print "id tolerance: %d" % tolerance
    for file in flist:
        frame_cnt += 1
        min_dist = 5.
        query_feat = np.load(file)
        for sample in sample_lib:
            dist = np.linalg.norm(query_feat-sample[0])
            if dist < min_dist:
                min_dist = dist
                match = sample[1]
        if get_video_name(file) != get_video_name(match):
            name_error += 1
            name_error_list.append((file, match, min_dist))
        elif get_frame_id(file) == get_frame_id(match):
            perfect += 1
        elif abs(get_frame_id(file) - get_frame_id(match)) <= tolerance:
            valid += 1
        else:
            id_error += 1
            id_error_list.append((file, match, min_dist))
            
        if frame_cnt % 100 == 0: 
            t1 = time.time()
            print "  %d:  %d %d %d %d  %.1f%%  %ds" % (frame_cnt, perfect, valid, name_error, id_error, (100*float(perfect+valid)/float(frame_cnt)), (t1-t0))

    t1 = time.time()
    score = float(perfect+valid)/float(frame_cnt)
    print "perfect: %d,   valid: %d" % (perfect, valid)
    print "name_error: %d,   id_error: %d" % (name_error, id_error)
    print "Score : %.1f %%" % (100*score)
    print("Total time: %d sec" % (t1-t0))
    
    result_csv = 'name_error_%s.csv' % mode
    print("Saving error results to %s..." % result_csv)
    f = open(result_csv, 'w')
    for item in name_error_list:
        string = "%s,%s,%f\n" % (item[0], item[1], item[2])
        f.write(string)
    f.flush()
    f.close()

    result_csv = 'id_error_%s.csv' % mode
    print("Saving error results to %s..." % result_csv)
    f = open(result_csv, 'w')
    for item in id_error_list:
        string = "%s,%s,%f\n" % (item[0], item[1], item[2])
        f.write(string)
    f.flush()
    f.close()
  
def thread_compare(tid, file_list, param, q, tolerance=75):
    print "Task [%2d] running, query range: %d ~ %d" %(tid, file_list[0], file_list[1])
    
    total = file_list[1] - file_list[0] + 1
    perfect    = 0
    valid      = 0
    name_error  = 0
    id_error = 0
    name_error_list = []
    id_error_list = []
    perfect_d  = 0.
    valid_d    = 0.
    name_error_d = 0.
    id_error_d   = 0.
    frame_cnt = 0
    frame_id = 0
    
    t0 = time.time()   
    for file in param.query_flist:
        frame_id += 1
        if frame_id < file_list[0]: continue
        if frame_id > file_list[1]: break
        
        frame_cnt += 1
        min_dist = 5.
        query_feat = np.load(file)
        for sample in param.sample_lib:
            dist = np.linalg.norm(query_feat-sample[0])
            if dist < min_dist:
                min_dist = dist
                match = sample[1]
        if get_video_name(file) != get_video_name(match):
            name_error += 1
            name_error_d += min_dist
            name_error_list.append((file, match, min_dist))
        elif get_frame_id(file) == get_frame_id(match):
            perfect += 1
            perfect_d += min_dist
        elif abs(get_frame_id(file) - get_frame_id(match)) <= tolerance:
            valid += 1
            valid_d += min_dist
        else:
            id_error += 1
            id_error_d += min_dist
            id_error_list.append((file, match, min_dist))
            
        if frame_cnt % 100 == 0: 
            t1 = time.time()
            print "  [%2d] %d|%d  %d %d %d %d  %.1f%%  %ds" % (tid, total, frame_cnt, perfect, valid, name_error, id_error, (100*float(perfect+valid)/float(frame_cnt)), (t1-t0))

    q.put([perfect,valid,name_error, id_error, perfect_d,valid_d,name_error_d, id_error_d])
    t1 = time.time()
    print "  [%2d] <END>: %d %d %d %d  %.1f%%  %ds" % (tid, perfect, valid, name_error, id_error, (100*float(perfect+valid)/float(frame_cnt)), (t1-t0))            
    
    result_csv = 'result_csv/error_name_%d.csv' % tid
    f = open(result_csv, 'w')
    for item in name_error_list:
        string = "%s,%s,%f\n" % (item[0], item[1], item[2])
        f.write(string)
    f.flush()
    f.close()

    result_csv = 'result_csv/error_id_%d.csv' % tid
    f = open(result_csv, 'w')
    for item in id_error_list:
        string = "%s,%s,%f\n" % (item[0], item[1], item[2])
        f.write(string)
    f.flush()
    f.close()


def rerange(list, order, topN):
    if order == (topN-1): return
    i = topN
    while (i > order):       
        list[i] = list[i-1]
        i -= 1
    #print list
    
def thread_compare_topN(tid, file_list, param, q, topN=1, tolerance=75):
    print "Task [%2d] running, query range: %d ~ %d" %(tid, file_list[0], file_list[1])
    
    total = file_list[1] - file_list[0] + 1
    perfect    = 0
    valid      = 0
    name_error  = 0
    id_error = 0
    name_error_list = []
    id_error_list = []
    perfect_d  = 0.
    valid_d    = 0.
    name_error_d = 0.
    id_error_d   = 0.
    frame_cnt = 0
    frame_id = 0
    
    t0 = time.time()   
    for file in param.query_flist:
        frame_id += 1
        if frame_id < file_list[0]: continue
        if frame_id > file_list[1]: break
        
        frame_cnt += 1
        min_dist = []
        for _ in range(topN+1): min_dist.append([5., ""])    # min -> larger
        query_feat = np.load(file)
        for sample in param.sample_lib:
            dist = np.linalg.norm(query_feat-sample[0])
            if dist >= min_dist[topN-1][0]: continue
            for n in range(topN): 
                if dist < min_dist[n]:
                    rerange(min_dist, n, topN)
                    min_dist[n] = [dist, sample[1]]
                    break
        
        for idx in range(topN):
            if get_video_name(file) == get_video_name(min_dist[idx][1]):
                if get_frame_id(file) == get_frame_id(min_dist[idx][1]):
                    perfect += 1
                    perfect_d += min_dist[idx][0]
                    break
                elif abs(get_frame_id(file) - get_frame_id(min_dist[idx][1])) <= tolerance:
                    valid += 1
                    valid_d += min_dist[idx][0]
                    break
                
            
        if frame_cnt % 100 == 0: 
            t1 = time.time()
            print "  [%2d] %d|%d  %d %d  %.1f%%  %ds" % (tid, total, frame_cnt, perfect, valid, (100*float(perfect+valid)/float(frame_cnt)), (t1-t0))

    q.put([perfect,valid,perfect_d,valid_d])
    t1 = time.time()
    print "  [%2d] <END>: %d %d  %.1f%%  %ds" % (tid, perfect, valid, (100*float(perfect+valid)/float(frame_cnt)), (t1-t0))            
   

    
def evaluate_multi(thread_num=2, model_id=0, topN=5):
    if model_id == 0:
        sample = os.path.join(my_root, 'train_refer_vggspoc')     
        query  = os.path.join(my_root, 'train_refer_crop_vggspoc')
    elif model_id == 1:
        sample = os.path.join(my_root, 'train_refer_hvr')     
        query  = os.path.join(my_root, 'train_refer_crop_hvr')
    elif model_id == 2:
        sample = os.path.join(my_root, 'train_refer_alexspoc')     
        query  = os.path.join(my_root, 'train_refer_crop_alexspoc')
    
    #sample = os.path.join(my_root, 'mini_feat_alexspoc')
    #query  = os.path.join(my_root, 'mini_crop_feat_alexspoc')
    
    param = eval_param()
    t0 = time.time()    
    flist = []
    fop.recursive_file(sample, flist)
    for sample_file in flist:
        param.sample_lib.append((np.load(sample_file), sample_file))
    print " --- Evaluator start ---"
    print "Sample_root : %s, frame number: %d" % (sample, len(flist))
    
    fop.recursive_file(query, param.query_flist)
    print "Query_root  : %s, frame number: %d" % (query, len(param.query_flist))

    task_list = []
    q = pop.queue()
    total = len(param.query_flist)
    if total <= thread_num: thread_num = total
    each  = total / thread_num
    for tid in range(thread_num):
        start = tid*each + 1
        end   = start + each - 1
        if tid+1 == thread_num: end = total
        task = pop.task(func=thread_compare_topN, args=(tid+1, (start,end), param, q, topN))
        task.run()
        task_list.append((tid+1, task))
    
    for i in range(thread_num):
        ret = q.get()
        param.perfect    += ret[0]
        param.valid      += ret[1]
        param.perfect_d  += ret[2]
        param.valid_d    += ret[3]

    for tid, task in task_list:
        task.close()
        print "Task [%d] exited." % tid        
    print "== All Tasks have exited. ==\n"
    
    score = float(param.perfect+param.valid)/total
    if param.perfect : param.perfect_d /= param.perfect
    if param.valid : param.valid_d /= param.valid
    #if param.name_error : param.name_error_d /= param.name_error
    #if param.id_error : param.id_error_d /= param.id_error
    print "sample_root: %s" % sample
    print "query_root : %s" % query
    print "TOP_N : %d" % topN
    print "perfect   :  %d (avg_dist=%.3f)   valid   :  %d (avg_dist=%.3f)" % (param.perfect, param.perfect_d, param.valid, param.valid_d)
    #print "name_error:  %d (avg_dist=%.3f)   id_error:  %d (avg_dist=%.3f)" % (param.name_error, param.name_error_d, param.id_error, param.id_error_d)
    print "Score     :  %.1f%%" % (100*score)
    t1 = time.time(); print "Total eval time: %d sec" % (t1-t0)       
  
def test_move_feat_file():
    import shutil
    base = '/home1/hvr2/crct_spoc/22'
    out_root = '/home1/hvr2/crct_spoc/train_query_vlad_feature'
    
    for file in os.listdir(base):
        fs = os.path.join(base, file)
        if os.path.isfile(fs):
            src = fs
            dst = os.path.join(out_root, fs.split('/')[-1])
            #print "111: ",src,dst;exit()
            shutil.move(src, dst)
        elif os.path.isdir(fs):
            for f in os.listdir(fs):
                src = os.path.join(fs, f)
                dst = os.path.join(out_root, f)
                #print "222: ",src,dst;exit()
                shutil.move(src, dst)


if __name__ == '__main__':
    '''
    sample = '/home1/hvr2/crct_new/train_refer_alex_feat'
    query  = '/home1/hvr2/crct_new/train_refer_crop_alex_feat'    
    evaluate(sample, query, mode='alex')

    sample = '/home1/hvr2/crct_new/train_refer_crow_feat'
    query  = '/home1/hvr2/crct_new/train_refer_crop_crow_feat'    
    evaluate(sample, query, mode='crow')
    '''
    #test_multi()
    #test_range()
    evaluate_multi(thread_num=20, model_id=0, topN=1)

