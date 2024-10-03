from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
from .predict import load_model, predict



app = FastAPI()

templates = Jinja2Templates(directory="templates")
MODEL = load_model()

@app.get("/ping")
async def ping() -> str:
    return "Hello, I am alive"

@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    try:
        image_data = await file.read()
        result = predict(MODEL, image_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)