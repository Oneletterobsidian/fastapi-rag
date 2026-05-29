from pydantic import BaseModel


class Image(BaseModel):
    url: str
    alt_text: str = ""


class ProductCreate(BaseModel):
    name: str
    price: float
    category: str
    in_stock: bool = True
    description: str | None = None


class ProductResponse(BaseModel):
    id: int
    name: str
    price: float
    category: str
    in_stock: bool
    description: str | None = None

    class Config:
        from_attributes = True


class ProductListResponse(BaseModel):
    total: int
    products: list[ProductResponse]

# ============================================
# 更新模型(PUT 用,所有字段必填)
# ============================================
class ProductUpdate(BaseModel):
    name: str
    price: float
    category: str
    in_stock: bool
    description: str | None = None

# ============================================
# PATCH 部分更新(所有字段都可选)
# ============================================
class ProductPatch(BaseModel):
    name: str | None = None
    price: float | None = None
    category: str | None = None
    in_stock: bool | None = None
    description: str | None = None