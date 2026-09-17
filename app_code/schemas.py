from pydantic import BaseModel
from typing import Optional

class UserRegister(BaseModel):
    username: str  
    email: str 
    password: str
    age: Optional[int] = None

class AskRequest(BaseModel):
    question: str