from werkzeug.exceptions import HTTPException
from werkzeug.sansio.response import Response

class QMatrixShapeMismatchError(HTTPException):
    code = 520
    
    def __init__(self, description: str | None = None, response: Response | None = None, **kwargs) -> None:
        description = f"The shape of the Q-matrix does not match the expected shape {kwargs['expected_shape']}, actual shape: {kwargs['actual_shape']}"
        super().__init__(description, response)