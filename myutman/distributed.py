from myutman.single_thread import StreamingAlgo
from myutman.exceptions import UnimplementedException

class DistributedStreamingAlgo(StreamingAlgo):
    def __init__(self, p, single_threads):
        super(DistributedStreamingAlgo, self).__init__(p)
        self.single_threads = single_threads

    def fuse(self, stats):
        raise UnimplementedException()

    def get_stat(self):
        return self.fuse([single.get_stat() for single in self.single_threads])

    def restart(self):
        for single in self.single_threads:
            single.restart()


class RoundrobinStreamingAlgo(DistributedStreamingAlgo):
    def __init__(self, p, single_threads):
        super(RoundrobinStreamingAlgo, self).__init__(p, single_threads)
        self.cur = 0

    def process_element(self, element, meta=None):
        self.single_threads[self.cur].process_element(element)
        self.cur = (self.cur + 1) % len(self.single_threads)


class DependentStreamingAlgo(DistributedStreamingAlgo):
    def __init__(self, p, single_threads):
        super(DependentStreamingAlgo, self).__init__(p, single_threads)
        self.cur = 0

    def process_element(self, element, meta=None):
        self.single_threads[hash(meta) % len(self.single_threads)].process_element(element)