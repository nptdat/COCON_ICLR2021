from pydantic import BaseModel


class GenerationRequest(BaseModel):
    model_id: str
    prompt_text: str
    context: str
    length: int

class GenerationResponse(BaseModel):
    generated_text: str
