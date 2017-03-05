"""
Common Data Primitives
"""

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
        self.label= None

    def __str__(self):
        ret = self.name + ': {'
        fields = [str(field) for field in [self.subset, self.label, self.duration, self.resolution, self.url, len(self.annotations)]]
        ret += ','.join(fields)
        ret += '}'
        return ret