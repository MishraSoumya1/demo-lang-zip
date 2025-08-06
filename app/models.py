from pydantic import BaseModel

class IncidentQuery(BaseModel):
    query: str
