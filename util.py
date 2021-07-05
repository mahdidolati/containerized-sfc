import itertools


class PopularityRecord:
    def __init__(self):
        self.pq = list()
        self.entry_finder = dict()
        self.removed = 'Removed'
        self.counter = itertools.count()

    def add_task(self, layer_id, layer_val):
        prev_val = 0
        if layer_id in self.entry_finder:
            prev_val = self.remove_task(layer_id)
        count = next(self.counter)
        entry = [prev_val - layer_val, count, layer_id]
        self.entry_finder[layer_id] = entry
        heapq.heappush(self.pq, entry)

    def remove_task(self, layer_id):
        prev_val = self.entry_finder[layer_id]
        entry = self.entry_finder.pop(layer_id)
        entry[-1] = self.removed
        return prev_val

    def pop_task(self):
        while self.pq:
            priority, count, task = heapq.heappop(self.pq)
            if task is not self.removed:
                del self.entry_finder[task]
                return task
        raise KeyError('pop from an empty priority queue')