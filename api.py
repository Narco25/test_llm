import importlib.util
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


_model_path = Path(__file__).with_name("model.py")
_model_spec = importlib.util.spec_from_file_location("model_wrapper", _model_path)
if _model_spec is None or _model_spec.loader is None:
    raise ImportError(f"Unable to load model wrapper from {_model_path}")
model = importlib.util.module_from_spec(_model_spec)
_model_spec.loader.exec_module(model)


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
    result_text = model.generate(request.prompt)
    return {"response": result_text}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)