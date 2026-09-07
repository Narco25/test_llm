from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model_wrapper import generate as generate_text


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://nikos-website-12z.pages.dev",
        "http://localhost:4321",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PromptRequest(BaseModel):
    prompt: str


@app.post("/generate")
def generate(request: PromptRequest) -> dict[str, str]:
    result_text = generate_text(request.prompt)
    return {"response": result_text}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)