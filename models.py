from sqlalchemy import Column, Integer, String, Float, Boolean, Text
from database import Base

# ============================================
# 商品表 (Product)
# ============================================
class Product(Base):
    __tablename__ = "products"          # 数据库里的表名

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    price = Column(Float, nullable=False)
    category = Column(String(50), nullable=False)
    in_stock = Column(Boolean, default=True)
    description = Column(Text, nullable=True)