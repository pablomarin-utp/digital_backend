import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base


DATABASE_URL = os.getenv("SUPABASE_DB_URL", "postgresql://user:pass@localhost:5432/dbname")

# Create a synchronous engine
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def init_db():
    # Import models so they are registered with SQLAlchemy
    try:
        from app.models import db_models  # noqa: F401
        Base.metadata.create_all(bind=engine)
    except Exception:
        # If models cannot be imported during startup, ignore here; they'll be imported later
        pass
