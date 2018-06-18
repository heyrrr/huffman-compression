"""
    >>> tree = HuffmanNode(None, HuffmanNode(None, HuffmanNode(None,\
    HuffmanNode(1), HuffmanNode(8)), HuffmanNode(2)), HuffmanNode(None,\
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(3), HuffmanNode(6)),\
    HuffmanNode(None, HuffmanNode(5), HuffmanNode(7))), HuffmanNode(4)))
    >>> number_nodes(tree)
    >>> tree.number
    6
    >>> tree.right.number
    5
    >>> tree.right.left.number
    4
"""
