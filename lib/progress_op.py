import os
from multiprocessing import Process,Pool,Queue,Lock

class task():
    '''
    Create one progress for task.
    '''
    def __init__(self, func, args):
        '''       
        func: entry func for task.
        args: args for main func, must be in tuple format. if there is only one arg A, should be writen as (A,).
        '''
        self.task = Process(target=func, args=args)
    
    def run(self):
        self.task.start()
        
    def close(self):
        self.task.join()
    
    
class queue():
    def __init__(self, size=100):
        self.q = Queue(size)

    def __exit__(self):    
        self.q.close()
        
    def get(self, timeout=None, block=True):
        '''
        timeout:  max seconds to wait for get(). None means wait for ever
        block  :  if set False, timeout is nonsense
        '''
        try: return self.q.get(block, timeout)
        except: return None
    
    def put(self, sth):
        self.q.put(sth)

        