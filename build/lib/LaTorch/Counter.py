# class for counting the occurrences of a set of symbols
class Counter():
    def __init__(self, open_symbols, close_symbols, start_values=None):
        self.open_symbols = open_symbols
        self.close_symbols = close_symbols
        
        if start_values == None:
            start_values = [0 for _ in open_symbols]
        else:
            self.start_values = start_values

        self.counters = [start_value for start_value in start_values] 

    def update(self, ch):
        for i in range(len(self.counters)):
            if ch == self.open_symbols[i]:
                self.counters[i] += 1
            elif ch == self.close_symbols[i]:
                self.counters[i] -= 1

    def is_zero(self):
        for i in range(len(self.counters)):
            if self.counters[i] != 0:
                return False
        return True
