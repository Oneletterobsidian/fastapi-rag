from sqlalchemy.orm import Session
from fastapi import HTTPException

from models import Product
import schemas


# ============================================
# 商品 CRUD 辅助函数
# ============================================

def get_product_by_id(db: Session, product_id: int) -> Product:
    """
    按 ID 查找商品,找不到则抛出 404
    """
    product = db.query(Product).filter(Product.id == product_id).first()
    if product is None:
        raise HTTPException(status_code=404, detail=f"商品 {product_id} 不存在")
    return product


def list_products(
    db: Session,
    keyword: str = "",
    category: str = "",
    min_price: float | None = None,
    max_price: float | None = None,
    in_stock_only: bool = False,
    limit: int = 100,
    skip: int = 0,
) -> tuple[int, list[Product]]:
    """
    带过滤和分页的商品列表查询
    返回: (总数, 商品列表)
    """
    query = db.query(Product)
    
    if keyword:
        query = query.filter(Product.name.contains(keyword))
    if category:
        query = query.filter(Product.category == category)
    if min_price is not None:
        query = query.filter(Product.price >= min_price)
    if max_price is not None:
        query = query.filter(Product.price <= max_price)
    if in_stock_only:
        query = query.filter(Product.in_stock == True)
    
    total = query.count()
    products = query.order_by(Product.id).offset(skip).limit(limit).all()
    
    return total, products


def create_product(db: Session, product_data: schemas.ProductCreate) -> Product:
    """
    创建新商品
    """
    new_product = Product(
        name=product_data.name,
        price=product_data.price,
        category=product_data.category,
        in_stock=product_data.in_stock,
        description=product_data.description,
    )
    db.add(new_product)
    db.commit()
    db.refresh(new_product)
    return new_product


def update_product(db: Session, product_id: int, product_data: schemas.ProductUpdate) -> Product:
    """
    完整更新商品(PUT)
    """
    product = get_product_by_id(db, product_id)    # 复用!
    
    product.name = product_data.name
    product.price = product_data.price
    product.category = product_data.category
    product.in_stock = product_data.in_stock
    product.description = product_data.description
    
    db.commit()
    db.refresh(product)
    return product


def delete_product(db: Session, product_id: int) -> None:
    """
    删除商品
    """
    product = get_product_by_id(db, product_id)    # 复用!
    db.delete(product)
    db.commit()

def patch_product(db: Session, product_id: int, product_data: schemas.ProductPatch) -> Product:
    """
    部分更新商品(PATCH)
    只更新传了值的字段
    """
    product = get_product_by_id(db, product_id)
    
    # 只更新非 None 的字段
    update_data = product_data.model_dump(exclude_unset=True)
    
    for field, value in update_data.items():
        setattr(product, field, value)
    
    db.commit()
    db.refresh(product)
    return product