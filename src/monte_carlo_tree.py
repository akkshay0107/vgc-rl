class MCTNode:
    def __init__(self, parent=None, state=None) -> None:
        self.parent = parent
        self.childs = []
        self.visits = 0
        self.value = 0.0
        self.state = state

    def __iter__(self):
        """
        Iterator for the node. Just iterates over the children of the node.
        """
        return iter(self.childs)


# TODO: implement the UCB1 algorithm for the MCTNode class
# Complete the implementation of the MCTNode class
# Create a dummy state and benchmark performance for the MCTNode
