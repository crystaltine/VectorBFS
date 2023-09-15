class Edge:
    def __init__(self, start, finish, weight) -> None:
        self.start = start
        self.finish = finish
        self.weight = weight

    def __eq__(self, other):
        return (self.start == other.start and self.finish == other.finish)
                        
    def __hash__(self) -> int:
        return hash((self.start, self.finish))