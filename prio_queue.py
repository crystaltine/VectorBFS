class PriorityQueueItem:
    def __init__(self, priority:float, value:any) -> None:
        self.value = value
        self.priority = priority

class PriorityQueue:
    def __init__(self) -> None:
        self.heap:list[PriorityQueueItem] = list([None])# create a blank first item so that we can start list at 1

    # adds an element with priority priority and value to the priority queue 
    def enqueue(self, priority:float, value:any):
        self.heap.append(PriorityQueueItem(priority, value))
        self._bubble_up(len(self.heap))

    # returns the value of the item at the front of the queue and removes the item
    def dequeue(self)->any:
        highest_pri = self.heap[1]
        last_item = self.heap.pop()

        # if the queue is not empty, move the bottom to the root and bubble down
        if not self.is_empty():  
            self.heap[1] = last_item # move last heap item to the top
            self._bubble_down(1)
        return highest_pri.value
      
    # returns the value of the item at the front of the queue without removing it
    def peek(self)->any:
        return self.heap[1].value

    def size(self)->int:
        return len(self.heap)-1

    def is_empty(self)->bool:
        return len(self.heap) == 1

    def change_priority(self, value, new_priority:int)->None:
        for i in range(1, len(self.heap)-1):
            if self.heap[i].value == value:
                break
        
        old_pri = self.heap[i].priority
        self.heap[i].priority = new_priority

        # bubble up and bubble down will return immediately if we are going the wrong way, so call both
        if new_priority > old_pri:
            self._bubble_down(i)
        else:
            self._bubble_up(i) 

    def clear(self):
        self.heap = list([None])      

    #####################
    # Helper Functions
    #####################
    def _get_parent_index(self, index)->int:
        return index//2

    def _get_first_child_index(self, index)->int:
        left_index = index*2
        if  left_index < len(self.heap):
            return left_index
        return None
    
    def _get_second_child_index(self, index)->int:
        right_index = index*2+1
        if right_index < len(self.heap):
            return right_index
        return None

    def _bubble_up(self, index:int)->None:
        if index == 1: # we are already at the root, so don't bubble
            return
        parent_index = self._get_parent_index(index)
        current_priority = self.heap[index].priority
        if current_priority < self.heap[parent_index].priority:
            self._swap_nodes(index, parent_index)
            self._bubble_up(parent_index)
              
    def _bubble_down(self, index:int)->None:
        left_index = self._get_first_child_index(index)
        if left_index is None: # no children if index is not in the heap, so we're done
            return
        
        # otherwise, it means we need to swap and continue to bubble down
        swap_index = left_index # start by assuming we will swap with the first child
        
        # if the right child exists and is smaller than the left (has higher priority), 
        # make that the one to swap with
        right_index = self._get_second_child_index(index)
        if right_index is not None and (self.heap[right_index].priority < self.heap[left_index].priority):  
            swap_index = right_index

        # if the one to swap (child with the the highest pri) is higher pri, swap the values and continue to bubble down
        if self.heap[index].priority > self.heap[swap_index].priority:
            self._swap_nodes(swap_index, index)
            self._bubble_down(swap_index)

    def _swap_nodes(self, index1, index2):
        self.heap[index1], self.heap[index2] = self.heap[index2], self.heap[index1]