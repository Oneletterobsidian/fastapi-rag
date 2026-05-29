import bcrypt
from datetime import datetime, timedelta, timezone
from jose import jwt, JWTError
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

# ============================================
# 配置
# ============================================
SECRET_KEY = "你的密钥-生产环境一定要改成随机字符串"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# OAuth2 密码流(让 Swagger UI 有"登录按钮")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# ============================================
# 密码相关(直接用 bcrypt,不走 passlib)
# ============================================
def hash_password(password: str) -> str:
    """把明文密码哈希成不可逆字符串"""
    password_bytes = password.encode("utf-8")
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode("utf-8")

def verify_password(plain: str, hashed: str) -> bool:
    """验证密码是否匹配"""
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))

# ============================================
# Token 相关
# ============================================
def create_access_token(username: str) -> str:
    """生成 JWT token"""
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": username,
        "exp": expire,
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str) -> str:
    """解码 JWT,返回用户名;失败抛 401"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Token 无效")
        return username
    except JWTError:
        raise HTTPException(status_code=401, detail="Token 无效或已过期")

# ============================================
# 依赖注入函数 - 给受保护的接口用
# ============================================
def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    """从请求 header 拿 token,验证,返回用户名"""
    return decode_token(token)