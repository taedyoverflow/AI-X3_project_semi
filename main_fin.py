from asyncio import locks
from anyio import Lock
import face_recognition
from fastapi import FastAPI, Form, Request, Depends, WebSocket, WebSocketDisconnect, logger,status, UploadFile, File
import cv2
from fastapi.staticfiles import StaticFiles
import numpy as np
from starlette.templating import Jinja2Templates
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from database import SessionLocal, engine
import models 
from sqlalchemy.orm import Session
from starlette.responses import RedirectResponse
import os
import glob ## 추가
import io
from fastapi.responses import JSONResponse


models.Base.metadata.create_all(bind=engine)
templates = Jinja2Templates(directory="templates")

app = FastAPI()
UPLOAD_DIRECTORY = "static/image"

abs_path = os.path.dirname(os.path.realpath(__file__))
app.mount("/static", StaticFiles(directory=f"{abs_path}/static"))

def get_db():
    db = SessionLocal()
    try: 
        yield db
    finally:
        db.close()

@app.get("/")
async def home(req: Request, db: Session = Depends(get_db)):
    users = db.query(models.User).all()
    return templates.TemplateResponse("index.html", { "request": req, "users": users })

@app.post("/add")
def add_user(req: Request, image: UploadFile = File(...), user_name: str = Form(...), db: Session = Depends(get_db)):
    
    # 업로드된 이미지 파일의 경로
    file_path = os.path.join(UPLOAD_DIRECTORY, user_name + ".jpg")
    with open(file_path, "wb") as file_object:
            file_object.write(image.file.read())

    new_user = models.User(user_name=user_name, user_image="static/image/" + user_name + ".jpg")
    db.add(new_user)

    db.commit()
    url = app.url_path_for("home")
    return RedirectResponse(url=url, status_code=status.HTTP_303_SEE_OTHER)

@app.get("/delete/{user_id}")
def add(req: Request, user_id: int, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.user_id == user_id).first()
    os.remove(user.user_image)
    db.delete(user)
    db.commit()
    url = app.url_path_for("home")
    return RedirectResponse(url=url, status_code=status.HTTP_303_SEE_OTHER)

def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.25): ## 추가
    return list(face_recognition.face_distance(known_face_encodings, face_encoding_to_check) <= tolerance)






@app.get("/testify") ## 추가
async def testify(req: Request, db: Session = Depends(get_db)):
    return templates.TemplateResponse("testify.html", { "request": req})

@app.post("/verify_image/") #추가
async def verify_image(image: UploadFile = File(...)):
    db = SessionLocal()
    all_images = db.query(models.User).all()
    file_list = [user.user_image for user in all_images]

    try:
        # 클라이언트에서 전송한 이미지의 얼굴 인코딩 계산
        image_data = await image.read()
        img = face_recognition.load_image_file(io.BytesIO(image_data))
        face_encoding_to_check = face_recognition.face_encodings(img)[0]

        # 데이터베이스 이미지들의 얼굴 인코딩 계산
        known_face_encodings = []
        for file in file_list:
            image_path = file
            image = face_recognition.load_image_file(image_path)
            face_encoding = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(face_encoding)

        # 얼굴 유사도 비교
        results = compare_faces(known_face_encodings, face_encoding_to_check)
        if any(results):
            return JSONResponse(content={"result": "PASS"})
        else:
            return JSONResponse(content={"result": "Not PASS"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/detect")
async def read_root(request: Request):
    return templates.TemplateResponse("detect.html", {"request": request})

async def websocket_handler(websocket: WebSocket):
    await websocket.accept()
    video_capture = cv2.VideoCapture(0)

    db = SessionLocal()
    all_images = db.query(models.User).all()
    file_list = [user.user_image for user in all_images]
    file_names = [user.user_name for user in all_images]

    known_face_encodings = []

    for file_name in file_list:
        # 이미지 파일을 불러와서 얼굴 인코딩 수행
        image = face_recognition.load_image_file(file_name)
        face_encoding = face_recognition.face_encodings(image)[0]
        # 사용자 이름과 얼굴 인코딩을 각 리스트에 추가
        known_face_encodings.append(face_encoding)

    try:
        while True:
            ret, frame = video_capture.read()
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # Initialize variables for comparison
                name = "denied"
                min_distance = 0.5  # Set a threshold for face distance

                # Compare face encoding with known faces
                for known_encoding, known_name in zip(known_face_encodings, file_names):
                    face_distance = face_recognition.face_distance([known_encoding], face_encoding)[0]
                    if face_distance < min_distance:
                        min_distance = face_distance
                        name = "access"

                face_names.append(name)

            for (top, right, bottom, left), name in zip(face_locations, face_names):
                if name == "access":
                    cv2.rectangle(frame, (left * 4, top * 4), (right * 4, bottom * 4), (0, 255, 0), 2)
                else:
                    cv2.rectangle(frame, (left * 4, top * 4), (right * 4, bottom * 4), (0, 0, 255), 2)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left * 4 + 6, bottom * 4 - 6), font, 1.0, (255, 255, 255), 1)

            _, jpeg = cv2.imencode('.jpg', frame)
            await websocket.send_bytes(jpeg.tobytes())

    except WebSocketDisconnect:
        pass
    finally:
        video_capture.release()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket_handler(websocket)

@app.post("/save-image")
async def add(fileName: str = Form(...), image: UploadFile = File(...)):
    save_path = os.path.join(UPLOAD_DIRECTORY, f"{fileName}.jpg")  # 이미지의 저장 경로 설정
    os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)  # 디렉토리가 없는 경우 생성
    with open(save_path, "wb") as f:
        f.write(await image.read())
    return {"filename": fileName}