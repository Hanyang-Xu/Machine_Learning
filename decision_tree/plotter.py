from graphviz import Digraph

class DecisionTreePlotter:
    def __init__(self, tree, feature_names=None, label_names=None) -> None:
        self.tree = tree
        self.feature_names = feature_names
        self.label_names = label_names
        self.graph = Digraph('Decision Tree')

    def _build(self, dt_node):
        # 判断是否为内部节点
        if dt_node.children:
            # 获取当前特征名称
            feature_name = self.feature_names[dt_node.feature_index] if self.feature_names else str(dt_node.feature_index)
            
            # 创建当前节点
            self.graph.node(str(id(dt_node)), label=feature_name, shape='box')

            # 遍历当前节点的子节点
            for feature_value, dt_child in dt_node.children.items():
                # 递归构建子树
                self._build(dt_child)
                
                # 获取当前特征值对应的标签
                label = str(feature_value)
                if self.feature_names:
                    d_value = self.feature_names[dt_node.feature_index].get('value_names', {})
                    label = d_value.get(feature_value, label)
                
                # 创建从当前节点到子节点的边
                self.graph.edge(str(id(dt_node)), str(id(dt_child)), label=label, fontsize='10')
        else:
            # 叶子节点的标签
            label = self.label_names[dt_node.value] if self.label_names else str(dt_node.value)
            self.graph.node(str(id(dt_node)), label=label, shape='ellipse')

    def plot(self):
        """构建并展示决策树"""
        self._build(self.tree)
        self.graph.view()