# RamanSPy: Next-Gen Web Application

This project has been refactored from a Streamlit script into a modern Full-Stack Web Application for premium UI/UX and scalability.

## Architecture

- **Frontend**: Next.js 14 (App Router), TailwindCSS, Framer Motion, Recharts.
- **Backend**: FastAPI (Python), Uvicorn, RamanSPy.

## Getting Started

### Prerequisites
- Node.js 18+
- Python 3.9+
- access to terminal

### Running the Application

Double-click or run the start script:
```bash
./run_app.sh
```

Or manually:

1. **Backend**:
   ```bash
   cd backend
   pip install -r requirements.txt
   uvicorn main:app --reload --port 8000
   ```

2. **Frontend**:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

Open [http://localhost:3000](http://localhost:3000) to view the application.

## Project Structure

- `frontend/src/app`: Next.js pages and layouts (Dashboard, Upload, Process).
- `frontend/src/components`: Reusable UI components (Sidebar, Charts).
- `backend/app`: FastAPI application logic.
- `backend/app/services`: Business logic for spectral processing.
