from myutman.exceptions import UnimplementedException

class StreamingAlgo:

    def __init__(self, p):
        self.p = p

    def process_element(self, element, meta=None):
        raise UnimplementedException()

    def get_stat(self):
        raise UnimplementedException()

    def test(self):
        raise UnimplementedException()

    def restart(self):
        pass
