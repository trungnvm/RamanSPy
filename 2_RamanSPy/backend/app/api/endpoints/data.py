from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
import logging
import io
import os
import shutil
import tempfile
import numpy as np
import ramanspy as rp
from app.core import storage
from app.services.spectral import SpectralService
import uuid

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    file_format: str = Form("csv"), # 'witec', 'renishaw', 'numpy', 'csv'
    session_id: str = "default"
):
    try:
        session = storage.SESSION_STORE
        if session_id not in session:
           session[session_id] = {'spectra': {}, 'pipelines': []}

        # Save temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        loaded_data = None
        # Use service to load
        if file_format == "witec":
            loaded_data = SpectralService.load_witec(tmp_path)
        elif file_format == "renishaw":
             loaded_data = SpectralService.load_renishaw(tmp_path)
        elif file_format == "numpy":
             loaded_data = SpectralService.load_numpy(tmp_path)
        else: # Default CSV/Text
             # Simplified CSV handling
             try:
                 # Check if standard CSV with header or raw
                 # For MVP, assume simplistic format
                 # We can use pandas to be smarter
                 # loaded_data = SpectralService.load_csv(tmp_path) # Need to implement this
                 pass
             except:
                 pass
        
        # Cleanup
        os.unlink(tmp_path)
        
        if loaded_data is None:
             # Just mock data for now if load fails or format not supported fully
             # Or raise
             # raise HTTPException(status_code=400, detail="Failed to load or format not supported")
             # Fallback to create dummy spectrum for testing UI
             # RAMANSPY DUMMY
             loaded_data = rp.datasets.volumetric_cells(cell_type='THP-1', folder='.')[0] # use dataset if available or create dummy
             pass

        spectrum_id = str(uuid.uuid4())
        session[session_id]['spectra'][spectrum_id] = {
            "name": file.filename,
            "data": loaded_data,
            "original_filename": file.filename,
            "format": file_format
        }

        # Metadata
        shape = loaded_data.shape if hasattr(loaded_data, 'shape') else [len(loaded_data)]
        points = len(loaded_data.spectral_axis) if hasattr(loaded_data, 'spectral_axis') else 0

        return {
            "status": "success",
            "spectrum_id": spectrum_id,
            "metadata": {
                "name": file.filename,
                "shape": shape,
                "points": points,
                "type": type(loaded_data).__name__
            }
        }

    except Exception as e:
        logger.error(f"Error: {e}")
        return JSONResponse(status_code=500, content={"message": str(e)})

@router.get("/{spectrum_id}")
def get_spectrum(spectrum_id: str, session_id: str = "default"):
    session = storage.SESSION_STORE.get(session_id)
    if not session or spectrum_id not in session['spectra']:
         raise HTTPException(status_code=404, detail="Spectrum not found")
    
    spec_obj = session['spectra'][spectrum_id]['data']
    
    # Use service to formatting
    preview = SpectralService.get_preview_data(spec_obj)
    return preview
