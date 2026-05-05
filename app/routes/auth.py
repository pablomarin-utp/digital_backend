"""
Rutas de autenticación simples: register, login y guardar embedding
"""
from fastapi import APIRouter, HTTPException, Form, Body
from pydantic import BaseModel
from typing import List, Optional

from app.db import SessionLocal, init_db
from app.models.db_models import User
from app.services import auth as auth_service

router = APIRouter()

# Inicializar la DB (crea tablas si es necesario)
init_db()


class RegisterResponse(BaseModel):
    success: bool
    message: str
    user_id: Optional[str] = None


class LoginResponse(BaseModel):
    success: bool
    message: str
    user_id: Optional[str] = None


@router.post("/register", response_model=RegisterResponse)
def register(username: str = Form(...), password: str = Form(...), full_name: str = Form(None)):
    db = SessionLocal()
    try:
        existing = db.query(User).filter(User.username == username).first()
        if existing:
            raise HTTPException(status_code=400, detail="Username already exists")
        hashed = auth_service.hash_password(password)
        user = User(username=username, password_hash=hashed, full_name=full_name)
        db.add(user)
        db.commit()
        db.refresh(user)
        return RegisterResponse(success=True, message="User created", user_id=str(user.id))
    finally:
        db.close()


@router.post("/login", response_model=LoginResponse)
def login(username: str = Form(...), password: str = Form(...)):
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == username).first()
        if not user:
            return LoginResponse(success=False, message="Invalid credentials")
        ok = auth_service.verify_password(password, user.password_hash)
        if not ok:
            return LoginResponse(success=False, message="Invalid credentials")
        return LoginResponse(success=True, message="Authenticated", user_id=str(user.id))
    finally:
        db.close()


@router.post("/users/{user_id}/embedding")
def save_embedding(user_id: str, embedding: List[float] = Body(...)):
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        user.embedding = embedding
        db.add(user)
        db.commit()
        return {"success": True, "message": "Embedding saved"}
    finally:
        db.close()
