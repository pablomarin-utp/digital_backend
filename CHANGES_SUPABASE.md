Resumen de cambios para integración con Supabase (autenticación propia)

Hechos principales:
- Se añadió persistencia usando Postgres (Supabase) vía SQLAlchemy (síncrono).
- Se implementaron endpoints de registro (/api/v1/auth/register), login (/api/v1/auth/login)
  y almacenamiento de embeddings (/api/v1/auth/users/{user_id}/embedding).
- El frontend ahora usa las rutas del backend para registrar y loguear usuarios.

Archivos añadidos / modificados (resumen archivo por archivo):

1) digital_backend/requirements.txt
   - Se añadieron dependencias: sqlalchemy, psycopg2-binary, passlib[bcrypt], python-dotenv

2) digital_backend/app/db.py (nuevo)
   - Configura la conexión SQLAlchemy a la DB usando la variable de entorno SUPABASE_DB_URL.
   - Exporta SessionLocal y Base y la función init_db() para crear tablas.

3) digital_backend/app/models/db_models.py (nuevo)
   - Define el modelo User (id, username, password_hash, full_name, created_at, embedding).

4) digital_backend/app/services/auth.py (nuevo)
   - Funciones hash_password y verify_password usando passlib (bcrypt).

5) digital_backend/app/routes/auth.py (nuevo)
   - Implementa endpoints:
     - POST /api/v1/auth/register (Form: username, password, full_name)
     - POST /api/v1/auth/login (Form: username, password)
     - POST /api/v1/auth/users/{user_id}/embedding (JSON body: array de floats)
   - Usa SessionLocal para acceder a la DB y auth_service para hashear/verificar contraseñas.

6) digital_backend/main.py (modificado)
   - Incluye el router auth en /api/v1/auth.

7) digital_backend/.env.example (nuevo)
   - Variables de entorno: SUPABASE_DB_URL, SUPABASE_SERVICE_ROLE_KEY, SUPABASE_JWT_SECRET (vacío por ahora)

8) digital_backend/sql/create_users.sql (nuevo)
   - SQL para crear la tabla users en Supabase. Guardar / ejecutar en SQL Editor de Supabase.

9) digital_frontend/src/services/auth.ts (nuevo)
   - Cliente simple para llamar a register/login/saveEmbedding del backend.

10) digital_frontend/src/App.tsx (modificado)
   - Añadido UI y lógica mínima para registrar/login y guardar user_id en localStorage.

11) digital_frontend/.env.example (verificado)
   - Asegura que VITE_API_URL apunte a http://localhost:8000/api/v1 para desarrollo.

Notas operativas y pasos para poner en marcha:
1. En Supabase, en SQL Editor, ejecutar digital_backend/sql/create_users.sql para crear la tabla users.
2. En el proyecto backend, crear un archivo .env con las variables (ver .env.example):
   SUPABASE_DB_URL=postgresql://<user>:<pass>@<host>:5432/<db>
   SUPABASE_SERVICE_ROLE_KEY=<service-role-key>
3. Instalar dependencias del backend:
   pip install -r requirements.txt
4. Ejecutar el backend (desde la carpeta digital_backend):
   uvicorn main:app --reload
5. En el frontend (digital_frontend):
   - Verificar VITE_API_URL en .env.local (por ejemplo http://localhost:8000/api/v1)
   - npm install
   - npm run dev
6. Probar registro/login usando la UI o curl (ejemplos abajo):

Ejemplos curl:
Registro:
curl -X POST -F "username=test" -F "password=secret" -F "full_name=Test" http://localhost:8000/api/v1/auth/register

Login:
curl -X POST -F "username=test" -F "password=secret" http://localhost:8000/api/v1/auth/login

Guardar embedding (ejemplo):
curl -X POST -H "Content-Type: application/json" -d "[0.1,0.2,0.3]" http://localhost:8000/api/v1/auth/users/<user_id>/embedding

Notas finales:
- La autenticación basada en JWT la implementaremos en otra rama como solicitaste.
- Actualmente el backend maneja hashing de contraseñas con bcrypt (passlib).
- Las imágenes no se envían ni se guardan en Supabase en esta etapa; en su lugar puedes enviar vectores (embeddings) al endpoint correspondiente.
