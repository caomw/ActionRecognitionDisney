"""
A helper class to store and load RGB frames and optical flow
"""

import os
import cPickle as pickle


class CacheManager:
    def __init__(self, root):
        self.__root = root
        os.system("mkdir -p {0}".format(root))
        os.system("mkdir -p {0}/framestack".format(root))
        os.system("mkdir -p {0}/flowstack".format(root))

    def dump(self, data, videoname, type):
        videoname = videoname.split('/')[-1].split('.')[0]
        path = "{0}/{1}/{2}".format(self.__root, type, videoname)
        if not os.path.exists(path):
            pickle.dump(data, open(path, 'wb'))
        else:
            return

    def load(self, videoname, type):
        videoname = videoname.split('/')[-1].split('.')[0]
        path = "{0}/{1}/{2}".format(self.__root, type, videoname)
        if os.path.exists(path):
            return pickle.load(open(path, 'rb'))
        else:
            return None


