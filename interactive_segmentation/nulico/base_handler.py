from abc import ABC, abstractmethod, ABCMeta


class BaseHandler(ABC):
    @abstractmethod
    def handle(self, image):
        pass

