"""
Code for compressing and decompressing using Huffman compression.
"""

from nodes import HuffmanNode, ReadNode


# ====================
# Helper functions for manipulating bytes


def get_bit(byte, bit_num):
    """ Return bit number bit_num from right in byte.

    @param int byte: a given byte
    @param int bit_num: a specific bit number within the byte
    @rtype: int

    >>> get_bit(0b00000101, 2)
    1
    >>> get_bit(0b00000101, 1)
    0
    """
    return (byte & (1 << bit_num)) >> bit_num


def byte_to_bits(byte):
    """ Return the representation of a byte as a string of bits.

    @param int byte: a given byte
    @rtype: str

    >>> byte_to_bits(14)
    '00001110'
    """
    return "".join([str(get_bit(byte, bit_num))
                    for bit_num in range(7, -1, -1)])


def bits_to_byte(bits):
    """ Return int represented by bits, padded on right.

    @param str bits: a string representation of some bits
    @rtype: int

    >>> bits_to_byte("00000101")
    5
    >>> bits_to_byte("101") == 0b10100000
    True
    """
    return sum([int(bits[pos]) << (7 - pos)
                for pos in range(len(bits))])


# ====================
# Functions for compression


def make_freq_dict(text):
    """ Return a dictionary that maps each byte in text to its frequency.

    @param bytes text: a bytes object
    @rtype: dict{int,int}

    >>> d = make_freq_dict(bytes([65, 66, 67, 66]))
    >>> d == {65: 1, 66: 2, 67: 1}
    True
    """
    d = {}
    for byte in text:
        if byte in d:
            d[byte] += 1
        else:
            d[byte] = 1
    return d


def huffman_tree(freq_dict):
    """ Return the root HuffmanNode of a Huffman tree corresponding
    to frequency dictionary freq_dict.

    @param dict(int,int) freq_dict: a frequency dictionary
    @rtype: HuffmanNode

    >>> freq = {2: 6, 3: 4}
    >>> t = huffman_tree(freq)
    >>> result1 = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> result2 = HuffmanNode(None, HuffmanNode(2), HuffmanNode(3))
    >>> t == result1 or t == result2
    True
    """
    queue = HuffmanQueue()
    for byte in freq_dict:
        queue.add(HuffmanNode(freq_dict[byte], None, HuffmanNode(byte)))
    while len(queue) >= 2:
        two = queue.least_two()
        left, right = two[0], two[1]
        if right.left is None and left.left is not None:
            right, left = left, right
        if left.left is None:
            if right.left is None:
                queue.add(HuffmanNode(left.symbol + right.symbol,
                                      left.right, right.right))
            else:
                temp = HuffmanNode(left.symbol + right.symbol,
                                   left.right, right)
                temp.right.symbol = None
                queue.add(temp)
        else:
            temp = HuffmanNode(left.symbol + right.symbol,
                               left, right)
            temp.right.symbol = None
            temp.left.symbol = None
            queue.add(temp)
    if len(queue) == 1:
        result = queue.least()

        result.symbol = None
        return result


class HuffmanQueue:
    """ A queue object that cooperates with Huffman tree

    """

    def __init__(self):
        """Initialize a HuffmanQueue.
        """
        self.queue = []

    def add(self, node):
        """Add a HuffmanNode to the queue and place it
        according to its frequency.
        :param node: HuffmanNode
        :return: NoneType
        """
        if len(self) == 0:
            self.queue.append(node)
        elif node.symbol >= self.queue[-1].symbol:
            self.queue.append(node)
        else:
            for _ in range(len(self.queue)):
                if self.queue[_].symbol > node.symbol:
                    self.queue.insert(_, node)
                    break

    def least_two(self):
        """Return the first two nodes in queue.

        :return: HuffmanNode
        """
        if len(self) > 1:
            return self.queue.pop(0), self.queue.pop(0)

    def __len__(self):
        return len(self.queue)

    def least(self):
        """Return the first node in queue.

        :return: HuffmanNode
        """
        if len(self) == 1:
            return self.queue.pop(0)


def get_codes(tree):
    """ Return a dict mapping symbols from tree rooted at HuffmanNode to codes.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: dict(int,str)

    >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> d = get_codes(tree)
    >>> d == {3: "0", 2: "1"}
    True
    """
    d = {}
    if tree.symbol is not None:
        return [tree.symbol, '']
    if tree.left is not None or tree.right is not None:
        left = get_codes(tree.left)
        right = get_codes(tree.right)
        if isinstance(left, list):
            left[1] = '0' + left[1]
            d.update({left[0]: left[1]})
        else:
            for item in left:
                left[item] = '0' + left[item]
            d.update(left)
        if isinstance(right, list):
            right[1] = '1' + right[1]
            d.update({right[0]: right[1]})
        else:
            for item in right:
                right[item] = '1' + right[item]
            d.update(right)
    return d


def number_nodes(tree):
    """ Number internal nodes in tree according to postorder traversal;
    start numbering at 0.

    @param HuffmanNode tree:  a Huffman tree rooted at node 'tree'
    @rtype: NoneType

    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(None, HuffmanNode(9), HuffmanNode(10))
    >>> tree = HuffmanNode(None, left, right)
    >>> number_nodes(tree)
    >>> tree.left.number
    0
    >>> tree.right.number
    1
    >>> tree.number
    2
    """
    counter(0)
    numbering(tree)


_ = 0


def counter(i=None):
    """Counter that keeps track of number.

    :param i: int
    :return: int
    """
    global _
    if i is not None:
        _ = -1
    else:
        _ += 1
    return _


def numbering(tree):
    """Helper function to number the nodes in tree.

    :param tree: HuffmanNode
    :return: int
    """
    if tree is not None:
        numbering(tree.left)
        numbering(tree.right)
        if not tree.is_leaf():
            tree.number = counter()
            return _


def avg_length(tree, freq_dict):
    """ Return the number of bits per symbol required to compress text
    made of the symbols and frequencies in freq_dict, using the Huffman tree.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @param dict(int,int) freq_dict: frequency dictionary
    @rtype: float

    >>> freq = {3: 2, 2: 7, 9: 1}
    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(9)
    >>> tree = HuffmanNode(None, left, right)
    >>> avg_length(tree, freq)
    1.9
    """
    codes = get_codes(tree)
    total_len = 0
    compressed_len = 0
    for item in codes:
        compressed_len += (len(codes[item]) * freq_dict[item])
    for item in freq_dict:
        total_len += freq_dict[item]
    return compressed_len / total_len


def generate_compressed(text, codes):
    """ Return compressed form of text, using mapping in codes for each symbol.

    @param bytes text: a bytes object
    @param dict(int,str) codes: mappings from symbols to codes
    @rtype: bytes

    >>> d = {0: "0", 1: "10", 2: "11"}
    >>> text = bytes([1, 2, 1, 0])
    >>> result = generate_compressed(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111000']
    >>> text = bytes([1, 2, 1, 0, 2, 2, 2, 2 ,1])
    >>> result = generate_compressed(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111001', '11111111', '00000000']
    """
    result = []
    blist = []
    for byte in text:
        if result == []:
            result.extend(generate_compressed_helper1(codes[byte]))
        elif len(result[-1]) + len(codes[byte]) <= 8:
            result[-1] += codes[byte]
        else:
            tail = generate_compressed_helper2(codes[byte], result[-1])
            result.pop(-1)
            result.extend(tail)
    for item in result:
        blist.append(bits_to_byte(item))
    return bytes(blist)


def generate_compressed_helper1(byte):
    """A helper function.

    :type byte: str
    :return: list
    """
    result = []
    if len(byte) < 8:
        result.append(byte)
    else:
        temp = byte
        while temp != '':
            result.append(temp[:8])
            temp = temp[8:]
    return result


def generate_compressed_helper2(byte, tail):
    """A helper function

    :param byte: str
    :param tail: str
    :return: list
    """
    result = [tail]
    for _ in range(len(byte)):
        if len(result[-1]) < 8:
            result[-1] += byte[_]
        else:
            result.append(byte[_:_ + 8])
            temp = byte[_ + 8:]
            while temp != '':
                result.append(temp[:8])
                temp = temp[8:]
            break
    return result


def tree_to_bytes(tree):
    """ Return a bytes representation of the tree rooted at tree.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: bytes

    The representation should be based on the postorder traversal of tree
    internal nodes, starting from 0.
    Precondition: tree has its nodes numbered.

    >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2]
    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(5)
    >>> tree = HuffmanNode(None, left, right)
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2, 1, 0, 0, 5]
    """
    result = ordering_bytes(tree)
    return bytes(result)


def ordering_bytes(tree):
    """Return a bytes representation of the tree rooted at tree.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: bytes

    The representation should be based on the postorder traversal of tree
    internal nodes, starting from 0.
    Precondition: tree has its nodes numbered.
    """
    result = []
    if tree.left is not None:
        result.extend(ordering_bytes(tree.left))
    if tree.right is not None:
        result.extend(ordering_bytes(tree.right))
    if tree.left is not None:
        if tree.left.is_leaf():
            result.append(0)
            result.append(tree.left.symbol)
        else:
            result.append(1)
            result.append(tree.left.number)
    if tree.right is not None:
        if tree.right.is_leaf():
            result.append(0)
            result.append(tree.right.symbol)
        else:
            result.append(1)
            result.append(tree.right.number)
    return result


def num_nodes_to_bytes(tree):
    """ Return number of nodes required to represent tree (the root of a
    numbered Huffman tree).

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: bytes
    """
    return bytes([tree.number + 1])


def size_to_bytes(size):
    """ Return the size as a bytes object.

    @param int size: a 32-bit integer that we want to convert to bytes
    @rtype: bytes

    >>> list(size_to_bytes(300))
    [44, 1, 0, 0]
    """
    # little-endian representation of 32-bit (4-byte)
    # int size
    return size.to_bytes(4, "little")


def compress(in_file, out_file):
    """ Compress contents of in_file and store results in out_file.

    @param str in_file: input file whose contents we want to compress
    @param str out_file: output file, where we store our compressed result
    @rtype: NoneType
    """
    with open(in_file, "rb") as f1:
        text = f1.read()
    freq = make_freq_dict(text)
    tree = huffman_tree(freq)
    codes = get_codes(tree)
    number_nodes(tree)
    print("Bits per symbol:", avg_length(tree, freq))
    result = (num_nodes_to_bytes(tree) + tree_to_bytes(tree) +
              size_to_bytes(len(text)))
    result += generate_compressed(text, codes)
    with open(out_file, "wb") as f2:
        f2.write(result)


# ====================
# Functions for decompression


def generate_tree_general(node_lst, root_index):
    """ Return the root of the Huffman tree corresponding
    to node_lst[root_index].

    The function assumes nothing about the order of the nodes in the list.

    @param list[ReadNode] node_lst: a list of ReadNode objects
    @param int root_index: index in the node list
    @rtype: HuffmanNode

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 1, 1, 0)]
    >>> generate_tree_general(lst, 2)
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(10, None, None), \
HuffmanNode(12, None, None)), \
HuffmanNode(None, HuffmanNode(5, None, None), HuffmanNode(7, None, None)))
    """
    d = {}
    for _ in range(len(node_lst)):
        d[_] = HuffmanNode(None, HuffmanNode(symbol=node_lst[_].l_data) if
                           node_lst[_].l_type == 0 else None,
                           HuffmanNode(symbol=node_lst[_].r_data) if
                           node_lst[_].r_type == 0 else None)
    for _ in range(len(node_lst)):
        if node_lst[_].l_type == 1:
            d[_].left = d[node_lst[_].l_data]
        if node_lst[_].r_type == 1:
            d[_].right = d[node_lst[_].r_data]
    return d[root_index]


def generate_tree_postorder(node_lst, root_index):
    """ Return the root of the Huffman tree corresponding
    to node_lst[root_index].

    The function assumes that the list represents a tree in postorder.

    @param list[ReadNode] node_lst: a list of ReadNode objects
    @param int root_index: index in the node list
    @rtype: HuffmanNode

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 0, 1, 0)]
    >>> generate_tree_postorder(lst, 2)
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(5, None, None), \
HuffmanNode(7, None, None)), \
HuffmanNode(None, HuffmanNode(10, None, None), HuffmanNode(12, None, None)))
    """
    huff_nodes = []
    for _ in range(len(node_lst)):
        offset = 1
        if node_lst[_].r_type == 0:
            huff_nodes.append(HuffmanNode(right=HuffmanNode(
                symbol=node_lst[_].r_data)))
        elif node_lst[_].r_type == 1:
            huff_nodes.append(HuffmanNode(right=huff_nodes[_ - offset]))
            offset += 1
        if node_lst[_].l_type == 0:
            huff_nodes[-1].left = HuffmanNode(symbol=node_lst[_].l_data)
        elif node_lst[_].l_type == 1:
            huff_nodes[-1].left = huff_nodes[_ - offset]
    return huff_nodes[root_index]


def generate_uncompressed(tree, text, size):
    """ Use Huffman tree to decompress size bytes from text.

    @param HuffmanNode tree: a HuffmanNode tree rooted at 'tree'
    @param bytes text: text to decompress
    @param int size: how many bytes to decompress from text.
    @rtype: bytes
    """
    d1 = get_codes(tree)
    d = {}
    for item in d1:
        d[d1[item]] = item
    temp = ''
    uncompressed = []
    for byte in text:
        for _ in range(7, -1, -1):
            if d.get(temp) is None:
                temp += str(get_bit(byte, _))
            if d.get(temp) is not None:
                uncompressed.append(d[temp])
                temp = ''
    uncompressed = uncompressed[:size]
    return bytes(uncompressed)


def bytes_to_nodes(buf):
    """ Return a list of ReadNodes corresponding to the bytes in buf.

    @param bytes buf: a bytes object
    @rtype: list[ReadNode]

    >>> bytes_to_nodes(bytes([0, 1, 0, 2]))
    [ReadNode(0, 1, 0, 2)]
    """
    lst = []
    for i in range(0, len(buf), 4):
        l_type = buf[i]
        l_data = buf[i + 1]
        r_type = buf[i + 2]
        r_data = buf[i + 3]
        lst.append(ReadNode(l_type, l_data, r_type, r_data))
    return lst


def bytes_to_size(buf):
    """ Return the size corresponding to the
    given 4-byte little-endian representation.

    @param bytes buf: a bytes object
    @rtype: int

    >>> bytes_to_size(bytes([44, 1, 0, 0]))
    300
    """
    return int.from_bytes(buf, "little")


def uncompress(in_file, out_file):
    """ Uncompress contents of in_file and store results in out_file.

    @param str in_file: input file to uncompress
    @param str out_file: output file that will hold the uncompressed results
    @rtype: NoneType
    """
    with open(in_file, "rb") as f:
        num_nodes = f.read(1)[0]
        buf = f.read(num_nodes * 4)
        node_lst = bytes_to_nodes(buf)
        # use generate_tree_general or generate_tree_postorder here
        tree = generate_tree_general(node_lst, num_nodes - 1)
        size = bytes_to_size(f.read(4))
        with open(out_file, "wb") as g:
            text = f.read()
            g.write(generate_uncompressed(tree, text, size))


# ====================
# Other functions

def improve_tree(tree, freq_dict):
    """ Improve the tree as much as possible, without changing its shape,
    by swapping nodes. The improvements are with respect to freq_dict.

    @param HuffmanNode tree: Huffman tree rooted at 'tree'
    @param dict(int,int) freq_dict: frequency dictionary
    @rtype: NoneType

    >>> left = HuffmanNode(None, HuffmanNode(99), HuffmanNode(100))
    >>> right = HuffmanNode(None, HuffmanNode(101), \
    HuffmanNode(None, HuffmanNode(97), HuffmanNode(98)))
    >>> tree = HuffmanNode(None, left, right)
    >>> freq = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
    >>> improve_tree(tree, freq)
    >>> avg_length(tree, freq)
    2.31
    """
    l = []
    for item in freq_dict:
        l.append((freq_dict[item], item))
    l.sort()
    if tree is not None:
        remain = improve_left(tree.left, l)
        improve_right(tree.right, remain)


def improve_left(tree, freq_list):
    """Improve the left branch

    :param tree: HuffmanNode
    :return: list
    """
    if tree is not None:
        remaining = improve_left(tree.left, freq_list)
        remaining = improve_right(tree.right, remaining)
        if tree.is_leaf():
            tree.symbol = remaining[0][1]
            remaining.pop(0)
            return remaining
    return freq_list


def improve_right(tree, freq_list):
    """Improve the left branch

    :param tree: HuffmanNode
    :return: list
    """
    if tree is not None:
        remaining = improve_left(tree.left, freq_list)
        remaining = improve_right(tree.right, remaining)
        if tree.is_leaf():
            tree.symbol = remaining[-1][1]
            remaining.pop(-1)
            return remaining
    return freq_list


if __name__ == "__main__":
    import python_ta

    python_ta.check_all(config="huffman_pyta.txt")
    # TODO: Uncomment these when you have implemented all the functions
    import doctest
    doctest.testmod()

    import time

    mode = input("Press c to compress or u to uncompress: ")
    if mode == "c":
        fname = input("File to compress: ")
        start = time.time()
        compress(fname, fname + ".huf")
        print("compressed {} in {} seconds."
              .format(fname, time.time() - start))
    elif mode == "u":
        fname = input("File to uncompress: ")
        start = time.time()
        uncompress(fname, fname + ".orig")
        print("uncompressed {} in {} seconds."
              .format(fname, time.time() - start))
