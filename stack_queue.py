from collections import deque
from copy import deepcopy
from typing import Deque, Generic, Tuple, TypeVar

T = TypeVar("T")

class Stack(Generic[T]):
    def __init__(self) -> None:
        self.stack:Deque = deque()

    def push(self, item)->None:
        self.stack.append(item)

    def pop (self)->T:
        return self.stack.pop()

    def peek(self)->T:
        if len(self.stack)>0:
            return self.stack[-1]
        else:
            return None

    def is_empty(self) -> bool:
        return len(self.stack) == 0

class Queue(Generic[T]):
    def __init__(self) -> None:
        self.queue:Deque = deque()

    def enqueue(self, item)->None:
        self.queue.append(item)
        
    def enqueue_all(self, items)->None:
        self.queue.extend(items)

    def dequeue(self)->T:
        return self.queue.popleft()

    def peek(self)->T:
        if len(self.queue)>0:
            return self.queue[0]
        else:
            return None

    def is_empty(self) -> bool:
        return len(self.queue) == 0