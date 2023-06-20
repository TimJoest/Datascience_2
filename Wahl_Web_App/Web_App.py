from fastapi import FastAPI
from typing import Optional, Annotated, List

app = FastAPI()

class user():
    name: str
    lastname: str
    password: str
    email: str
    
    status: bool
    
class toping():
    name: str
    tags: List[str]
    description: str
    voters: List[str]
    upvotes: int
    downvotes: int
    abstentions: int
    status: bool
    


@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
async def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}

