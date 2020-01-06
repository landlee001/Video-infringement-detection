import os
import sys
import glob
import shutil

# Return True if there is file or file link, while returning False if dir or not exist.
def is_file(file):
    return os.path.isfile(file)

    
# Return True if there is dir or dir link, while returning False if file or not exist.
def is_dir(dir):
    return os.path.isdir(dir)    


def mkdir(dir):
    """
    Make dir like `mkdir -p` in bash
    """
    try:
        os.makedirs(dir)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(dir):
            pass
        else:
            raise

def copy(src, dst):
    shutil.copyfile(src, dst)            
            
def move(src, dst):
    shutil.move(src, dst)
            
def recursive_file(base, file_list, depth=-1):
    """
    Recursive Traversal for finding all files.
    
    base     :  In. top base directory with recursive subdirectories and files.
    file_list:  Out. file names with absolute path.
    depth   :  subdirectory depth. =0 meas not recursive; -1 means deepest recursive
    """
    for file in os.listdir(base):
        fs = os.path.join(base, file)
        if os.path.isfile(fs):
            file_list.append(fs)
        elif os.path.isdir(fs):
            if depth:
                depth -= 1
                recursive_file(fs, file_list, depth)

            
def recursive_dir(base, dir_list, depth=-1):
    """
    Recursive Traversal for finding all directories.
    
    base    :  In. top base directory with recursive subdirectories and files.
    dir_list:  Out. file names with absolute path.
    depth   :  subdirectory depth. =0 meas not recursive; -1 means deepest recursive
    """
    for file in os.listdir(base):
        fs = os.path.join(base, file)
        if os.path.isdir(fs):
            dir_list.append(fs)
            if depth:
                depth -= 1
                recursive_dir(fs, dir_list, depth)

            
def parse_csv(csv_name, member_num=3):
	a, b, c = ([] for i in range(member_num))
	csv_lines = open(csv_name, "r").readlines()
	#print('csv_line number: ', len(csv_lines))
	for i in range(len(csv_lines)):
		line = csv_lines[i]
		elems = line.rstrip().split(',')
		if len(elems) == member_num:
			txt = elems[0]
			if len(txt) != 0:
				a.append(txt)
			else:
				print('[ERR] a is null!')
			txt = elems[1]
			if len(txt) != 0:
				b.append(txt)
			else:
				print('[ERR] b is null!')
			txt = elems[1]
			if len(txt) != 0:
				c.append(txt)
			else:
				print('[ERR] c is null!')
		else:
			print('[ERR] line element numbers are not %d!!!' % member_num)
	#print('entry number: ', len(a))
	return a, b, c