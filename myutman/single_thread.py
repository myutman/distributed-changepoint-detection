from myutman.exceptions import UnimplementedException
import numpy as np

class StreamingAlgo:

    def __init__(self, p):
        self.p = p

    def process_element(self, element):
        raise UnimplementedException()

    def get_stat(self):
        raise UnimplementedException()

    def test(self):
        raise UnimplementedException()

    def restart(self):
        pass
