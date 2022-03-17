import os
import uuid
from shutil import copy2

#prefix_pathfiles = '/home/snow/github/land/lib/mimo_tools/data/results.bkp/'
#posfix_pathfiles = '/study/model.study.sqlite'


prefix_pathfiles = '/home/snow/github/land/lib/mimo_tools/data/dataset/results/'
scenes_dir = os.listdir(prefix_pathfiles)
posfix_pathfiles = '/study/model.paths.t001_01.r002.p2m'
posfix_pathgain = '/study/model.pg.t001_01.r002.p2m'

for scene_file in scenes_dir:
    pathfile1 = prefix_pathfiles + str(scene_file) + posfix_pathfiles
    pathfile2 = prefix_pathfiles + str(scene_file) + posfix_pathgain
    path2create = '/home/snow/github/land/lib/mimo_tools/data/dataset/textfiles/' + str(scene_file) + '/'
    dest_dir1 = path2create + '/model.paths.t001_01.r002.p2m'
    dest_dir2 = path2create + '/model.pg.t001_01.r002.p2m'
    #current_file1 = None
    try:
        current_file2 = open(pathfile2)
        #os.mkdir(path2create)
        copy2(pathfile2, dest_dir2)
    except IOError:
        print("File not accessible: ", pathfile2)
    #finally:
    #    current_file1.close()

    #dest_dir2 = path2create + posfix_pathgain
    #current_file2 = None
    #try:
    #    #current_file2 = open(pathfile2)
    #    os.mkdir(path2create)
    #    #copy2(pathfile2, dest_dir2)
    #except IOError:
    #    print("File not accessible: ", pathfile2)
    #finally:
    #    current_file2.close()
