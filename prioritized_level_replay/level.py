class BaseLevel:
    """Base class for a training level."""
    def __init__(self, id: str) -> None:
        self.id = id

    def __str__(self) -> str:
        return str(self.id)

    def __repr__(self) -> str:
        return self.__str__()

class Level(BaseLevel):
    """A training level."""
    pass