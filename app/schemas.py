from pydantic import BaseModel, EmailStr
from datetime import date as DateType, time as TimeType


class ChatRequest(BaseModel):
    session_id: str
    question: str


class BookingDetails(BaseModel):
    name: str | None = None
    email: EmailStr | None = None
    date: DateType | None = None
    time: TimeType | None = None
    confirmed: bool = False
