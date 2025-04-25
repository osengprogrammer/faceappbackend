from fastapi import FastAPI, UploadFile, File, HTTPException
from datetime import date, datetime
from sqlalchemy.orm import Session

from . import models, utils
from .. import database
import os
import cv2
import numpy as np
app = FastAPI()
models.Base.metadata.create_all(database.engine)

@app.post("/register/")
async def register(name: str, file: UploadFile = File(...)):
    emb = utils.get_embedding(await file.read())
    if emb is None:
        raise HTTPException(400, "No face detected")
    np.save(f"embeddings/{name}.npy", emb)  # 1 file per user
    return {"status": "registered"}

@app.post("/attendance/")
async def attendance(file: UploadFile = File(...),
                     db: Session = database.get_db()):
    # 1) Liveness: require blink
    frame = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
    if not utils.detect_blink(frame):
        raise HTTPException(400, "Blink not detected")

    # 2) Face recognition
    emb = utils.get_embedding(await file.read())
    if emb is None:
        raise HTTPException(400, "No face detected")

    # 3) Compare to stored embeddings
    for fname in os.listdir("embeddings"):
        known = np.load(f"embeddings/{fname}")
        dist = np.linalg.norm(known - emb)
        if dist < 0.6:
            user_id = fname[:-4]
            break
    else:
        raise HTTPException(404, "User not recognized")

    # 4) Prevent duplicate check-in/out
    today = date.today()
    rec = db.query(models.Attendance).filter_by(user_id=user_id, date=today).first()
    now = datetime.now()
    if not rec:
        rec = models.Attendance(user_id=user_id, date=today, check_in=now)
        db.add(rec); db.commit()
        return {"status": "checked in", "time": now}
    elif not rec.check_out:
        rec.check_out = now; db.commit()
        return {"status": "checked out", "time": now}
    else:
        return {"status": "already completed", "time": rec.check_out}
