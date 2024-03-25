from email.policy import default
from sqlalchemy import Column, Integer, LargeBinary, String
from database import Base

class User(Base):
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True)
    user_name = Column(String(100))
    user_image = Column(String(1000))