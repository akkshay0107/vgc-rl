from __future__ import annotations
from collections.abc import Iterator


class MCTNode[Any]:
    def __init__(
        self, parent: MCTNode[Any] | None = None, state: Any | None = None
    ) -> None:
        self.parent: MCTNode[Any] | None = parent
        self.childs: list[MCTNode[Any]] = []
        self.visits: int = 0
        self.value: float = 0.0
        self.state: Any | None = state

    def __iter__(self) -> Iterator[MCTNode[Any]]:
        """
        Iterator for the node. Just iterates over the children of the node.
        """
        return iter(self.childs)


# TODO: implement the UCB1 algorithm for the MCTNode class
# Complete the implementation of the MCTNode class
# Create a dummy state and benchmark performance for the MCTNode
