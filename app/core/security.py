from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.core.config import settings
security = HTTPBearer()
def validate_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != settings.auth_token:
        raise HTTPException(status_code=403, detail="Unauthorized")
    return True