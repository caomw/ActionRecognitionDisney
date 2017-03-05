""" ActivityNet Data Loader
@Yu Mao
"""

import json
from common import *


class DataActivityNet:
    def __init__(self, annotation_file, frame_folders):
        """
        initialize data manager for ActivityNet
        :param annotation_file: annotation file of ActivityNet
        :param frame_folders: a dict recording frame_type: frame_folder mappings
        The frame_folder must contain #videos of subdirectories each named as the video name.
        And each subdirectory should contain the frames extracted from the corresponding video.
        """
        self.__annotation_file = annotation_file
        self.__frame_folders = frame_folders
        self.labels = None
        self.label_idx_table = None
        self.taxonomy = None
        self.label_hierarchy = None
        self.version = None
        self.video_meta = {}

    def init(self):
        """
        load annotation file
        :return:
        """
        data = json.load(open(self.__annotation_file, 'r'))
        # fetch version info
        self.version = data['version'].split()[1]
        # parse labels
        self.labels, self.label_idx_table = DataActivityNet.__parse_labels(data['taxonomy'])
        self.label_hierarchy = LabelHierarchy()
        self.label_hierarchy.build(self.labels)
        # parse data-set
        self.video_meta = DataActivityNet.__parse_database(data['database'])

    @staticmethod
    def __parse_labels(taxonomy):
        labels = [Label(item['nodeId'], item['nodeName'], item['parentId'], item['parentName']) for item in taxonomy]
        label_idx_table = {item['nodeName']: item['nodeId'] for item in taxonomy}
        labels.sort(key=lambda label: label.id)
        return labels, label_idx_table

    @staticmethod
    def __parse_database(database):
        return {name: DataActivityNet.__construct_video_meta(name, database[name]) for name in database.keys()}

    @staticmethod
    def __construct_video_meta(name, meta_info_description):
        meta = VideoMetaInfo()
        meta.name = name
        meta.duration = meta_info_description['duration']
        meta.subset = meta_info_description['subset']
        meta.resolution = meta_info_description['resolution']
        meta.url = meta_info_description['url']
        meta.annotations = []
        for annotation_desc in meta_info_description['annotations']:
            annotation = VideoMetaInfo.SegmentAnnotation()
            annotation.start = annotation_desc['segment'][0]
            annotation.end = annotation_desc['segment'][1]
            annotation.duration = annotation.end - annotation.start
            annotation.label = annotation_desc['label']
            meta.annotations.append(annotation)
        meta.label = meta.annotations[0].label if len(meta.annotations) != 0 else None
        return meta

if __name__ == '__main__':
    data_manager = DataActivityNet('acnet.json', None)
    data_manager.init()
    print(data_manager.label_hierarchy)
    for k, v in data_manager.video_meta.iteritems():
        print(v)



