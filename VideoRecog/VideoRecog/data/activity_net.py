""" ActivityNet Data Manager
@Yu Mao
"""

import json


class Label:
    """
    structure storing the label of a class
    """
    def __init__(self, id, name, parent_id, parent_name):
        self.name = name
        self.id = id
        self.parent_id = parent_id
        self.parent_name = parent_name

    def __str__(self):
        return '(' + self.name + ', ' + str(self.id) + ')'


class LabelNode:
    """
    A node representing a label in the taxonomy hierarchy
    """
    def __init__(self, label):
        self.label = label
        self.childs = []
        self.parent = None
        self.level = 0

    def add_child(self, node):
        self.childs.append(node)

    def set_parent(self, parent):
        self.parent = parent
        self.level = self.parent.level + 1

    def __str__(self):
        ret = '\t' * (self.level) + ' ' + str(self.label)
        ret += '\n'
        for child in self.childs:
            ret += str(child)
        return ret


class LabelHierarchy:
    """
    A tree representing the label hierarchy
    """
    def __init__(self):
        self.root = None

    def build(self, labels):
        """
        build the label hierarchy
        :param labels:
        :return:
        """
        i = 0
        labels = list(labels)
        while i < len(labels):
            self.__add_label(labels[i], labels)
            i += 1

    def __add_label(self, label, labels):
        if self.find_node(label.id) is not None:
            raise Exception('duplicated node added: ' + str(label))

        node = LabelNode(label)
        if label.parent_id is None:
            self.root = node
            return
        parent_node = self.find_node(label.parent_id)
        if parent_node is None:
            for i in range(len(labels)):
                if labels[i].id == label.parent_id:
                    parent_label = labels.pop(i)
                    parent_node = self.__add_label(parent_label, labels)
                    break
        if parent_node is None:
            raise Exception('node with id ' + str(label.parent_id) + ' not found.')
        parent_node.add_child(node)
        node.set_parent(parent_node)
        return node

    def find_node(self, id):
        """
        find a node by id
        :param id: the node id
        :return: the reference to the node
        """
        return self.__find_node_recursively(id, self.root)

    def __find_node_recursively(self, id, root):
        if root is None:
            return None
        elif root.label.id == id:
            return root
        elif len(root.childs) == 0:
            return None
        else:
            for child in root.childs:
                result = self.__find_node_recursively(id, child)
                if result is not None:
                    return result
            return None

    def __str__(self):
        return str(self.root)


class VideoMetaInfo:
    """
    a structure holding the meta info of a video
    """
    class SegmentAnnotation:
        def __init__(self):
            self.start = 0
            self.end = 0
            self.duration = 0
            self.label = None

    def __init__(self):
        self.name = None
        self.duration = None
        self.subset = None
        self.resolution = None
        self.url = None
        self.annotations = None
        self.label = None

    def __str__(self):
        ret = self.name + ': {'
        fields = [str(field) for field in [self.subset, self.label, self.duration, self.resolution, self.url, len(self.annotations)]]
        ret += ','.join(fields)
        ret += '}'
        return ret


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

    def label_idx_to_name(self, label_idx):
        """
        convert the label index to its name
        :param label_idx:
        :return: label name
        """
        assert (label_idx < len(self.labels))
        return self.labels[label_idx].name

    def get_num_classes(self):
        return len(self.labels)

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
