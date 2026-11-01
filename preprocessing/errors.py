class WrongTupleSize(Exception):
    def __init__(self, array_xy):
        super().__init__(array_xy)
        self.message = "ERROR: Tuple length error should be 2, "
        self.array_xy = len(array_xy)
    def __str__(self):
        return f"{self.message}{self.array_xy} are not same size with 2"

class ArrayLengthError(Exception):
    def __init__(self, array_x, array_y):
        super().__init__(array_x, array_y)
        self.message = "ERROR: Array length error "
        self.array_x = len(array_x)
        self.array_y = len(array_y)
    def __str__(self):
        return f"{self.message}{self.array_x} are not same with {self.array_y}"

class InvalidDimensionError(Exception):
    pass

class ArrayColumnError(Exception):
    pass