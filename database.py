from abc import ABC

class BaseDatabase(ABC):
    def __init__(self,connection_params):
        pass

    def connect(self):
        pass

    def close(self):
        pass