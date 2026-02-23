"""
VoiceForge Main Server - All endpoints in one place
"""

import os
import sys
import json
import asyncio

_APP_DIR = os.path.dirname(os.path.dirname(__file__))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response

from config import (
    APP_DIR, OUTPUT_DIR, MODEL_DIR,
    ensure_dir, get_config, save_config
)
from servers.routers import (
    tts_router,
    rvc_router,
    postprocess_router,
    asr_router,
    files_router,
    comfyui_router,
)
from util.clients import (
    is_rvc_server_available,
    is_postprocess_server_available,
    is_whisperasr_available,
    is_chatterbox_server_available,
    is_pocket_tts_server_available,
    get_shared_session,
    RVC_SERVER_URL,
)


app = FastAPI(title="VoiceForge API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
ui_path = os.path.join(APP_DIR, "ui")
if os.path.exists(ui_path):
    app.mount("/static", StaticFiles(directory=ui_path), name="static")

ensure_dir(OUTPUT_DIR)

# Register routers
app.include_router(tts_router)
app.include_router(rvc_router)
app.include_router(postprocess_router)
app.include_router(asr_router)
app.include_router(files_router)
app.include_router(comfyui_router)


# =============================================================================
# CORE ENDPOINTS - Config, voices, models, health
# =============================================================================

@app.get("/")
async def root():
    index_path = os.path.join(APP_DIR, "ui", "index.html")
    if os.path.exists(index_path):
        return FileResponse(
            index_path, 
            media_type="text/html",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
    return {"name": "VoiceForge API", "version": "2.0.0"}


@app.get("/favicon.ico")
async def favicon():
    icon_path = os.path.join(APP_DIR, "assets", "icon.ico")
    if os.path.exists(icon_path):
        return FileResponse(icon_path, media_type="image/x-icon")
    return Response(status_code=204)  # No content


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/api/config")
async def get_cfg():
    """Get config.json"""
    return get_config()


@app.post("/api/config")
async def set_cfg(request: Request):
    """Save to config.json"""
    data = await request.json()
    if save_config(data):
        return {"success": True}
    raise HTTPException(500, "Failed to save")


@app.get("/api/models")
async def get_models():
    """List RVC models"""
    models = []
    if os.path.exists(MODEL_DIR):
        for item in os.listdir(MODEL_DIR):
            path = os.path.join(MODEL_DIR, item)
            if os.path.isdir(path):
                if os.path.exists(os.path.join(path, "model.pth")):
                    models.append(item)
    return {"models": models}


@app.get("/api/custom-models")
async def get_custom_models():
    """List custom trained TTS models (Chatterbox)."""
    import httpx
    
    custom_models = {
        "chatterbox": []
    }
    
    # Query Chatterbox server for custom models
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            res = await client.get("http://0.0.0.0:8893/v1/models")
            if res.status_code == 200:
                data = res.json()
                custom_models["chatterbox"] = [m for m in data.get("models", []) if m.get("type") == "custom"]
                custom_models["chatterbox_current"] = data.get("current", "default")
    except Exception as e:
        custom_models["chatterbox_error"] = str(e)
    
    # Also check local directories if server is not available
    chatterbox_custom_dir = os.path.join(APP_DIR, "models", "chatterbox_custom")
    if os.path.exists(chatterbox_custom_dir) and not custom_models["chatterbox"]:
        for name in os.listdir(chatterbox_custom_dir):
            if os.path.isdir(os.path.join(chatterbox_custom_dir, name)):
                custom_models["chatterbox"].append({"name": name, "type": "custom", "path": os.path.join(chatterbox_custom_dir, name)})
    
    return custom_models


@app.get("/api/modules")
async def get_modules():
    """Service availability - returns instantly, services checked on-demand when used"""
    # All services are optional - don't block connection checking them
    # Availability is determined when features are actually used
    return {
        "rvc": True, # Will fail gracefully if not running when used
        "postprocess": True,
        "chatterbox": True,
        "pocket_tts": True,
        "kokoro": True,
        "asr": True,
    }


@app.get("/api/settings/workers")
async def get_workers():
    """Worker settings"""
    cfg = get_config()
    return {
        "max_workers": cfg.get("max_workers", os.cpu_count() or 4),
        "rvc_workers": cfg.get("rvc_workers", 2),
    }


@app.post("/api/settings/workers")
async def set_workers(request: Request):
    """Update worker settings"""
    data = await request.json()
    valid = {k: v for k, v in data.items() 
             if k in ["max_workers", "rvc_workers"]}
    
    session = get_shared_session()
    if "rvc_workers" in valid:
        try: session.post(f"{RVC_SERVER_URL}/workers", data={"num_workers": valid["rvc_workers"]}, timeout=30)
        except: pass
    
    save_config(valid)
    return {"success": True}


# =============================================================================
# WEBSOCKET
# =============================================================================

_ws_connections: list[WebSocket] = []

@app.websocket("/ws/progress")
async def ws_progress(websocket: WebSocket):
    await websocket.accept()
    _ws_connections.append(websocket)
    try:
        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=30)
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "heartbeat"})
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in _ws_connections:
            _ws_connections.remove(websocket)


async def broadcast_progress(message: dict):
    for ws in _ws_connections[:]:
        try:
            await ws.send_json(message)
        except:
            if ws in _ws_connections:
                _ws_connections.remove(ws)


# =============================================================================
# STARTUP/SHUTDOWN
# =============================================================================

@app.on_event("startup")
async def startup():
    print("VoiceForge API ready!")


@app.on_event("shutdown")
async def shutdown():
    print("Shutting down...")


if __name__ == "__main__":
    import argparse
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8888")))
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    uvicorn.run("main_server:app", host=args.host, port=args.port, reload=args.reload)
