

"""
building decision tree code from scratch
refer to https://www.youtube.com/watch?v=LDRbO9a6XPU
source code https://github.com/random-forests/tutorials/blob/master/decision_tree.py#L1

sklearn decision tree https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def value_counts(list):
    """Counts the number of each type of example in a dataset."""
    counts = {}  # a dictionary of label -> count.
    for item in list:
        # in our dataset format, the label is always the last column
        label = item
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

def is_numeric(value):
    """Test if a value is numeric."""
    return isinstance(value, int) or isinstance(value, float)

# class Question:
#     """ A question to partition a data set.
#         This class just records a 'column number' (e.g., 0 for Color) and a
#     'column value' (e.g., Green). The 'match' method is used to compare
#     the feature value in an example to the feature value stored in the
#     question. See the demo below.
#     """
#     def __init__(self, header, column, value) -> None:
#         self.header = header
#         self.column = column
#         self.value = value
    
#     def match(self, example):
#         val = example[self.column]
#         if is_numeric(val):
#             return val >= self.value
#         else:
#             return val == self.value
    
#     def __repr__(self) -> str:
        
#         condition = "=="
#         if is_numeric(self.value):
#             condition = ">="

#         return "Is %s %s %s?" % (
#             self.header[self.column], condition, str(self.value)
#         )

class Leaf:
    """A Leaf node classifies data.

    This holds a dictionary of class (e.g., "Apple") -> number of times
    it appears in the rows from the training data that reach this leaf.
    """

    def __init__(self, rows):
        self.predictions = value_counts(rows)

class Decision_Node:
    """ A decision node asks a question.

     This holds a reference to the question, and to the two child nodes.
    """
    def __init__(self, feature, branches) -> None:
        # self.question = question
        self.feature = feature
        self.branches = branches
        # self.true_branch = true_branch
        # self.false_branch = false_branch

class MyID3DecisionTreeClassifier:

    def __init__(self, criterion = 'entropy', min_samples_leaf = 1) -> None:
        """
        Args:
           criterion: {“gini”, “entropy”, “log_loss”}, default=”gini” The function to measure the quality of a split.
           min_samples_leaf: The minimum number of samples required to be at a leaf node.
        """
        self.criterion = criterion
        self.min_samples_leaf = min_samples_leaf
        self.root = None
        self.feature_names = []
    
    def _calculate_entropy(self, subset):
        """ calculating the entropy of passing subset
        """
        entropy = 0
        n = len(subset)
        value_cnts = value_counts(subset)

        # unique label value and count
        for v, c in value_cnts.items():
            p_xi = c / n
            entropy += -1 * (p_xi * np.log2(p_xi))
        
        return entropy

    
    def _entropy(self, X, Y, feature_name = None):
        """ calculating the entropy of sets, if feature is not none, calculating sum of entropy of the each subsets that split by feature

            Args: X is dataframe
        """
        n_rows = len(Y)
        entropy = 0
        labels = Y # take last columns as label

        if feature_name is None:
            return self._calculate_entropy(labels)
            
        feature_col = X.loc[:, feature_name]
        feature_counts = value_counts(feature_col)
        for v, c in feature_counts.items(): 
            # take all the labels under this unique feature value
            labels_on_feature = labels[feature_col == v]
            entropy += c / n_rows * self._calculate_entropy(labels_on_feature)

        return entropy

    def _info_gain(self, X, Y, feature_name):
       
        entropy = self._entropy(X, Y)
        entropy_on_feature = self._entropy(X, Y, feature_name)

        return entropy - entropy_on_feature


    def _find_best_spilt(self, X, Y):
        """
        Find the best question to ask by iterating over every feature based on different
        criterion
        """
        feature_names = self.feature_names
        max_gain = 0
        best_feature = None
        
        if self.criterion == "entropy":
            n = len(self.feature_names)

            # iterate all the features
            for i in range(n):
                feature_name = feature_names[i]
                gain = self._info_gain(X, Y, feature_name)
                if max_gain < gain:
                    max_gain = gain
                    best_feature = feature_name

        return (max_gain, best_feature)

    def _partition(self, X, best_feature):
        branches = {}
        feature_col = X.loc[:, best_feature]

        # 遍历所有的可用的行
        for i in feature_col.index:
            v = feature_col[i]
            row = X.loc[i, :]
            if v not in branches:
                branches[v] = [row]
            else:
                branches[v].append(row)

        return branches

    def _build_tree(self, X, Y):
        """
        recursion
        divide and conquer
        """

        values_count = value_counts(Y)
        if len(values_count) == 1: # only have one category, no need to split
            return Leaf(Y)

        # if don't enough rows, just use it as leaf
        if len(X) <= self.min_samples_leaf:
            return Leaf(Y)

        # find the useful feature to split data based on different criterion
        max_gain, best_feature = self._find_best_spilt(X, Y)

        if max_gain == 0:
            return Leaf(Y)

        # if we reach here, we have found a useful feature to partition on
        branches_rows = self._partition(X, best_feature)

        # recursively build the true branch
        branches = {}
        for value, rows in branches_rows.items():
            best_feature_col = X.loc[:, best_feature]
            y_on_value = Y[best_feature_col == value]
            branches[value] = self._build_tree(pd.DataFrame(rows), y_on_value)

        return Decision_Node(best_feature, branches)

    def fit(self, X, Y):
        self.feature_names = X.columns
        self.root = self._build_tree(X, Y)
    
    def predict(self, X):
        y_predict = []

        for i in X.index:
            predictions = self._find(X.loc[i, :], self.root)
            y_predict.append(list(predictions.items())[0][0])
        
        return y_predict

    # recursively find the leaf node (result) in the builded tree based on the o vector of x
    def _find(self, row, node):
        if isinstance(node, Leaf):
            return node.predictions

        best_feature = node.feature

        row_value = row[self.feature_names.get_loc(best_feature)]
        if node.branches[row_value] != None:
            return self._find(row, node.branches[row_value])
        return None
    
# 获取叶节点数目
def get_numL_leafs(node):
    num_leafs = 0
    if isinstance(node, Leaf):
        return 1

    for _, child in node.branches.items():
        num_leafs += get_numL_leafs(child)

    return num_leafs

def get_depth(node):
    max_depth = 0
    if isinstance(node, Leaf):
        return 1
    
    for _, child in node.branches.items():
        depth = get_numL_leafs(child)
        if max_depth < depth:
            max_depth = depth

    return max_depth + 1
    
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
        arrow_args = dict(arrowstyle="<-")
        createPlot.axl.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction', xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1])/2.0 + cntrPt[1]
    createPlot.axl.text(xMid, yMid, txtString)


def plotTree(myTree, parentPt, nodeTxt):
    decision_node = dict(boxstyle="sawtooth", fc="0.8")
    left_node = dict(boxstyle="round4", fc="0.8")

    numLeafs = get_numL_leafs(myTree)
    depth = get_depth(myTree)

    firstStr = list(myTree.branches.keys())[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decision_node)
    secondDict = myTree.branches[firstStr]

    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.branches:
        if isinstance(secondDict.branches[key], Decision_Node):
            plotTree(secondDict.branches[key], cntrPt, str(key))
        else: # leaf node
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict.branches[key], (plotTree.xOff, plotTree.yOff),cntrPt, left_node)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD


def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.axl = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(get_numL_leafs(inTree))
    plotTree.totalD = float(get_depth(inTree))

    plotTree.xOff = -0.5/plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()

if __name__ == "__main__":
    import os
    import sys
    print(os.getcwd())
    # setting path
    sys.path.append('./lab_id3_tree')

    data = pd.read_csv('./chapter4/lab_id3_tree/tennis.csv')

    # 将文字转换成数字
    # data['Outlook'] = data['Outlook'].map({"sunny": 0, "overcast": 1, "rain": 2})
    # data['Temperature'] = data['Temperature'].map({"cool": 0, "mild": 1, "hot": 2})
    # data['Humidity'] = data['Humidity'].map({"high": 10, "normal": 5})
    # data['Windy'] = data['Windy'].map({False: 0, True: 1})
    # data['Decision'] = data['Decision'].map({"n": 0, "p": 1})

    #define the X and y
    X = data.drop(['Decision'],axis=1)
    y = data.loc[:,'Decision']


    # print(X.loc[1, :])
    # # print(X.loc[1, :][:, 'Outlook'])

    # for i  in X.loc[1, :]:
    #     print(i)

    a = pd.DataFrame([X.loc[1, :], X.loc[2, :]])
    print(a.head())

    id3_tree = MyID3DecisionTreeClassifier()
    id3_tree.fit(X,y)

    y_predict = id3_tree.predict(pd.DataFrame([X.loc[0, :]]))
    print(y_predict)

   
    createPlot(id3_tree.root)
    print("end")