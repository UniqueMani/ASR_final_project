import json, time
from pathlib import Path
from typing import Generator

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.streaming.chunker import iter_wav_chunks
from backend.streaming.stabilizer import SubtitleStabilizer
from backend.dialect.predict import DialectPredictor
from backend.asr.engine import make_asr_engine

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(title="Dialect + Subtitle MVP")
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

@app.get("/", response_class=HTMLResponse)
def index():
    return (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")

def sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

@app.post("/api/stream")
async def stream_subtitles(
    audio: UploadFile = File(...),
    simulate_realtime: bool = Form(True),
    chunk_sec: float = Form(4.0),
    overlap_sec: float = Form(1.0),
):
    tmp_dir = BASE_DIR / "_tmp"
    tmp_dir.mkdir(exist_ok=True)
    tmp_path = tmp_dir / f"upload_{int(time.time()*1000)}_{audio.filename}"
    with tmp_path.open("wb") as f:
        f.write(await audio.read())

    asr_engine = make_asr_engine()
    # Let mock engine know the original filename (for transcript lookup)
    try:
        asr_engine.set_current_context(audio.filename)
    except Exception:
        pass

    dialect = DialectPredictor()
    stab = SubtitleStabilizer(stability_rounds=2)

    def gen() -> Generator[str, None, None]:
        try:
            yield sse_event("meta", {"filename": audio.filename, "engine": asr_engine.name, "simulate_realtime": simulate_realtime})
            for chunk in iter_wav_chunks(str(tmp_path), chunk_sec=float(chunk_sec), overlap_sec=float(overlap_sec)):
                dial_pred = dialect.predict_chunk(chunk["pcm"], chunk["sr"])
                partial_text = asr_engine.transcribe_pcm(chunk["pcm"], chunk["sr"])
                committed, partial = stab.update(partial_text)

                yield sse_event("partial", {
                    "t0": chunk["t0"], "t1": chunk["t1"],
                    "dialect": dial_pred,
                    "committed": committed,
                    "partial": partial
                })

                if simulate_realtime:
                    sleep_sec = max(0.0, (chunk["t1"] - chunk["t0"]) - float(overlap_sec))
                    time.sleep(min(1.0, sleep_sec))
            final_text = asr_engine.transcribe_file(str(tmp_path))
            yield sse_event("final", {"text": final_text})
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    return StreamingResponse(gen(), media_type="text/event-stream")

@app.post("/api/offline")
async def offline_transcribe(audio: UploadFile = File(...)):
    tmp_dir = BASE_DIR / "_tmp"
    tmp_dir.mkdir(exist_ok=True)
    tmp_path = tmp_dir / f"upload_{int(time.time()*1000)}_{audio.filename}"
    with tmp_path.open("wb") as f:
        f.write(await audio.read())

    asr_engine = make_asr_engine()
    dialect = DialectPredictor()

    try:
        dial_pred = dialect.predict_file(str(tmp_path), max_sec=6.0)
        text = asr_engine.transcribe_file(str(tmp_path))
        return JSONResponse({"engine": asr_engine.name, "dialect": dial_pred, "text": text})
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
