from fastapi import FastAPI, UploadFile, File, Request, HTTPException, Depends, Form
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from datetime import date, datetime
import numpy as np
import cv2
import os

from app import models, utils, database

app = FastAPI()

# CORS Setup
origins = [
    "http://localhost:5173",                       # Local Vite dev
    "http://192.168.1.9:5173",                     # LAN IP for mobile testing
    "https://e295-103-179-182-19.ngrok-free.app",  # Frontend ngrok URL
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize DB
models.Base.metadata.create_all(bind=database.engine)

# Global error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"message": "Unexpected error occurred", "detail": str(exc)},
    )

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")

# DB Dependency
def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.post("/register/")
async def register_user(
    name: str = Form(...),
    file: UploadFile = File(...),
):
    """
    Register a new user's face by saving their face embedding.
    """
    try:
        # Read and debug
        img_bytes = await file.read()
        print("Received image bytes:", len(img_bytes))

        # Get the face embedding
        emb = utils.get_embedding(img_bytes)
        if emb is None:
            raise HTTPException(status_code=400, detail="No face detected")

        # Save embedding
        os.makedirs("embeddings", exist_ok=True)
        emb_path = os.path.join("embeddings", f"{name}.npy")
        np.save(emb_path, emb)

        return {"status": "registered", "user": name}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during registration: {str(e)}")


@app.post("/attendance/")
async def mark_attendance(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Mark attendance based on a face and blink detection.
    """
    try:
        img_bytes = await file.read()
        frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

        # Step 1: Detect blink
        if not utils.detect_blink(frame):
            raise HTTPException(status_code=400, detail="Blink not detected")

        # Step 2: Get embedding
        emb = utils.get_embedding(img_bytes)
        if emb is None:
            raise HTTPException(status_code=400, detail="No face detected")

        # Step 3: Match embedding
        user_id = None
        for fname in os.listdir("embeddings"):
            if not fname.endswith(".npy"):
                continue
            known_emb = np.load(os.path.join("embeddings", fname))
            dist = np.linalg.norm(known_emb - emb)
            if dist < 0.6:
                user_id = fname[:-4]  # strip “.npy”
                break

        if not user_id:
            raise HTTPException(status_code=404, detail="User not recognized")

        # Step 4: Record attendance
        today = date.today()
        now = datetime.now()
        rec = db.query(models.Attendance).filter_by(user_id=user_id, date=today).first()

        if not rec:
            rec = models.Attendance(user_id=user_id, date=today, check_in=now)
            db.add(rec)
            db.commit()
            return {"status": "checked in", "user": user_id, "time": now}
        elif not rec.check_out:
            rec.check_out = now
            db.commit()
            return {"status": "checked out", "user": user_id, "time": now}
        else:
            return {"status": "already completed", "user": user_id, "time": rec.check_out}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during attendance marking: {str(e)}")
