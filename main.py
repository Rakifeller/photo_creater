import os
from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from fastapi.responses import JSONResponse
from .pipelines import generate_with_images, prepare_embeds, generate_with_embeds
from .storage import save_pil, save_embeds, load_embeds, to_b64
from .schemas import PrecomputeResponse, GenerateByIdRequest, GenerateResponse
from .config import API_KEY, DATA_DIR

app = FastAPI(title="AI Selfie Backend", version="0.1")

def _check_auth(x_api_key: str | None):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/precompute", response_model=PrecomputeResponse)
async def precompute_identity(
    x_api_key: str | None = Header(None),
    files: list[UploadFile] = File(...)
):
    _check_auth(x_api_key)
    refs = [await f.read() for f in files]
    embeds = prepare_embeds(refs)
    path, fn = save_embeds(embeds)
    # basitçe identity_id = dosya adı
    return {"identity_id": fn}

@app.post("/generate")
async def generate(
    x_api_key: str | None = Header(None),
    prompt: str = Form(...),
    steps: int = Form(30),
    guidance: float = Form(5.0),
    seed: int | None = Form(None),
    width: int = Form(1024),
    height: int = Form(1024),
    return_base64: bool = Form(True),
    files: list[UploadFile] = File(...)
):
    _check_auth(x_api_key)
    refs = [await f.read() for f in files]
    img = generate_with_images(prompt, refs, steps, guidance, seed, (width, height))
    path, _ = save_pil(img)
    if return_base64:
        return JSONResponse({"image_base64": to_b64(path)})
    return JSONResponse({"image_path": path})

@app.post("/generate/by-id", response_model=GenerateResponse)
async def generate_by_id(req: GenerateByIdRequest, x_api_key: str | None = Header(None)):
    _check_auth(x_api_key)
    embeds = load_embeds(os.path.join(DATA_DIR, req.identity_id))
    img = generate_with_embeds(
        req.prompt, embeds, req.steps, req.guidance, req.seed, (req.width, req.height)
    )
    path, _ = save_pil(img)
    if req.return_base64:
        return {"image_base64": to_b64(path)}
    return {"image_path": path}
