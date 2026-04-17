from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates

app = FastAPI()

# Point FastAPI to our templates folder
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def home(request: Request):
    # Loads the main application shell (Navbar, etc.)
    return templates.TemplateResponse(request=request, name="base.html")

@app.get("/pages/data_processing")
async def data_processing(request: Request):
    # HTMX will grab this and swap it in instantly
    return templates.TemplateResponse(request=request, name="data_processing.html")

@app.get("/pages/model_training")
async def model_training(request: Request):
    return templates.TemplateResponse(request=request, name="model_training.html")

@app.get("/pages/model_evaluation")
async def model_evaluation(request: Request):
    return templates.TemplateResponse(request=request, name="model_evaluation.html")
