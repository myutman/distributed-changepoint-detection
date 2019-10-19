from myutman.exceptions import UnimplementedException


class StreamingAlgo:

    def __init__(self):
        pass

    def process_element(self, element):
        raise UnimplementedException()

    def get_stat(self):
        raise UnimplementedException()


class WindowStreamingAlgo(StreamingAlgo):

    def __d(self, reference_window, sliding_window):
        pass

    def __init__(self, window_count):
        super().__init__()
        self.window_count = window_count
        self.reference_windows = [None] * window_count
        self.sliding_windows = [None] * window_count

    def process_element(self, element):
        pass

    def get_stat(self):
        return [self.__d(self.reference_windows[i], self.sliding_windows[i]) for i in range(self.window_count)]



if __name__ == '__main__':
    print('kek')
