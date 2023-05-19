from fastapi import Request
from fastapi.exceptions import RequestValidationError, HTTPException
from fastapi.exception_handlers import http_exception_handler as _http_exception_handler
from fastapi.exception_handlers import (
    request_validation_exception_handler as _request_validation_exception_handler,
)
from fastapi.responses import JSONResponse
from fastapi.responses import PlainTextResponse
from fastapi.responses import Response

async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return PlainTextResponse("Invalid request body format", status_code=422) 

# async def http_exception_handler(request: Request, exc: HTTPException):
#     if exc.status_code == 405:
#         return PlainTextResponse("Wrong method, please use POST method with Application/Json", status_code=405)
#     else:
#         return exc