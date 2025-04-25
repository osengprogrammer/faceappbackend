from datetime import date, datetime
from sqlalchemy import Column, String, Date, DateTime
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Attendance(Base):
    __tablename__ = 'attendance faceapp'
    user_id = Column(String, primary_key=True)
    date    = Column(Date,   primary_key=True)
    check_in  = Column(DateTime, nullable=False)
    check_out = Column(DateTime, nullable=True)
