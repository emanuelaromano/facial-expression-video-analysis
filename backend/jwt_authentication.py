import os
import time
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
import jwt
from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from http import HTTPStatus

router = APIRouter()
bearer = HTTPBearer(auto_error=False)

########################################################
# API Endpoints
########################################################

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
ADMIN_SECRET_KEY = os.getenv("ADMIN_SECRET_KEY")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 14  # 14 days

if not JWT_SECRET_KEY or not ADMIN_SECRET_KEY:
    raise RuntimeError("JWT_SECRET_KEY and ADMIN_SECRET_KEY must be set.")

########################################################
# Helpers
########################################################

def _now_ts() -> int:
    return int(time.time())

def create_access_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None,
) -> str:
    exp = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    payload = {
        **data,
        "iat": _now_ts(),
        "nbf": _now_ts(),
        "exp": int(exp.timestamp()),
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

def decode_token(token: str) -> Optional[Dict[str, Any]]:
    try:
        token = jwt.decode(
            token,
            JWT_SECRET_KEY,
            algorithms=[JWT_ALGORITHM],
        )
        return {"validated": True, "expired": False}
    except jwt.ExpiredSignatureError:
        return {"validated": False, "expired": True}
    except jwt.InvalidTokenError:
        return {"validated": False, "expired": False}
    except Exception as e:
        return {"validated": False, "expired": False}

def create_guest_token() -> str:
    return create_access_token({"sub": "guest", "type": "access"})

def create_admin_token() -> str:
    return create_access_token({"sub": "admin", "type": "admin"})

########################################################
# API Endpoints
########################################################

@router.get("/validate", status_code=HTTPStatus.OK)
def validate_token_endpoint(credentials: HTTPAuthorizationCredentials = Depends(bearer)):
    if not credentials or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    try:
        payload = decode_token(credentials.credentials)
        return payload
    except Exception as e:
        raise HTTPException(status_code=401, detail="Token validation failed: " + str(e))

@router.post("/create/guest", status_code=HTTPStatus.OK)
def create_guest_token_endpoint(credentials: HTTPAuthorizationCredentials = Depends(bearer)):
    if not credentials or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    if credentials.credentials != ADMIN_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Invalid admin secret key")
    return {"token": create_guest_token()}

@router.post("/create/admin", status_code=HTTPStatus.OK)
def create_admin_token_endpoint(credentials: HTTPAuthorizationCredentials = Depends(bearer)):
    if not credentials or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    if credentials.credentials != ADMIN_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Invalid admin secret key")
    return {"token": create_admin_token()}
