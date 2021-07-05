from pydantic import BaseModel


class GenerationRequest(BaseModel):
    prompt_text: str
    context: str
    length: int

class GenerationResponse(BaseModel):
    generated_text: str
