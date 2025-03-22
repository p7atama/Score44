#!/bin/bash

# URL repository GitHub
REPO_URL="https://github.com/score-technologies/score-vision.git"

# Meminta jumlah clone dari user
read -p "Masukkan jumlah clone yang diinginkan: " CLONE_COUNT

# Validasi input harus berupa angka
if ! [[ "$CLONE_COUNT" =~ ^[0-9]+$ ]]; then
    echo "Input harus berupa angka."
    exit 1
fi

# Clone repository utama
echo "Meng-clone repository ke score-vision..."
git clone "$REPO_URL" "score-vision"

# Hapus main.py lama dan buat yang baru
echo "Menghapus main.py lama dan menggantinya dengan yang baru..."
rm -f score-vision/miner/main.py

cat > score-vision/miner/main.py <<EOL
import os
from fastapi import FastAPI
from loguru import logger

from fiber.logging_utils import get_logger
from miner.core.models.config import Config
from miner.core.configuration import factory_config
from miner.dependencies import get_config
from miner.endpoints.soccer import router as soccer_router
from miner.endpoints.availability import router as availability_router
from fastapi.middleware.cors import CORSMiddleware

# Setup logging
logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI()

# Untuk Ngocok Gan
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add dependencies
app.dependency_overrides[Config] = get_config

# Include routers with their prefixes and tags
app.include_router(
    soccer_router,
    prefix="/soccer",
    tags=["soccer"]
)
app.include_router(
    availability_router,
    tags=["availability"]
)
EOL

echo "File main.py telah diperbarui."

rm -f score-vision/miner/endpoints/soccer.py
cat > score-vision/miner/endpoints/soccer.py <<EOL
import os
import json
import time
from typing import Optional, Dict, Any
import supervision as sv
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
import asyncio
from pathlib import Path
from loguru import logger

from fiber.logging_utils import get_logger
from miner.core.models.config import Config
from miner.core.configuration import factory_config
from miner.dependencies import get_config, verify_request, blacklist_low_stake
from sports.configs.soccer import SoccerPitchConfiguration
from miner.utils.device import get_optimal_device
from miner.utils.model_manager import ModelManager
from miner.utils.video_processor import VideoProcessor
from miner.utils.shared import miner_lock
from miner.utils.video_downloader import download_video

logger = get_logger(__name__)

CONFIG = SoccerPitchConfiguration()

# Global model manager instance
model_manager = None

def get_model_manager(config: Config = Depends(get_config)) -> ModelManager:
    global model_manager
    if model_manager is None:
        model_manager = ModelManager(device=config.device)
        model_manager.load_all_models()
    return model_manager

async def process_soccer_video(
    video_path: str,
    model_manager: ModelManager,
) -> Dict[str, Any]:
    """Process a soccer video and return tracking data."""
    start_time = time.time()
    
    try:
        video_processor = VideoProcessor(
            device=model_manager.device,
            cuda_timeout=10800.0,
            mps_timeout=10800.0,
            cpu_timeout=10800.0
        )
        
        if not await video_processor.ensure_video_readable(video_path):
            logger.error(f"Vidionya bego: {video_path}")
            raise HTTPException(
                status_code=400,
                detail="Video file is not readable or corrupted"
            )
        
        player_model = model_manager.get_model("player")
        pitch_model = model_manager.get_model("pitch")
        
        tracker = sv.ByteTrack()
        
        tracking_data = {"frames": []}
        
        async for frame_number, frame in video_processor.stream_frames(video_path):
            pitch_result = pitch_model(frame, verbose=False)[0]
            keypoints = sv.KeyPoints.from_ultralytics(pitch_result)
            
            player_result = player_model(frame, imgsz=1280, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(player_result)
            detections = tracker.update_with_detections(detections)
            
            # Convert numpy arrays to Python native types
            frame_data = {
                "frame_number": int(frame_number),  # Convert to native int
                "keypoints": keypoints.xy[0].tolist() if keypoints and keypoints.xy is not None else [],
                "objects": [
                    {
                        "id": int(tracker_id),  # Convert numpy.int64 to native int
                        "bbox": [float(x) for x in bbox],  # Convert numpy.float32/64 to native float
                        "class_id": int(class_id)  # Convert numpy.int64 to native int
                    }
                    for tracker_id, bbox, class_id in zip(
                        detections.tracker_id,
                        detections.xyxy,
                        detections.class_id
                    )
                ] if detections and detections.tracker_id is not None else []
            }
            tracking_data["frames"].append(frame_data)
            
            if frame_number % 100 == 0:
                elapsed = time.time() - start_time
                fps = frame_number / elapsed if elapsed > 0 else 0
                logger.info(f"Processed {frame_number} frames in {elapsed:.1f}s ({fps:.2f} fps)")
        
        processing_time = time.time() - start_time
        tracking_data["processing_time"] = processing_time
        
        total_frames = len(tracking_data["frames"])
        fps = total_frames / processing_time if processing_time > 0 else 0
        logger.info(
            f"Completed processing {total_frames} frames in {processing_time:.1f}s "
            f"({fps:.2f} fps) on {model_manager.device} device"
        )
        
        return tracking_data
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Video processing error: {str(e)}")

async def process_challenge(
    request: Request,
    config: Config = Depends(get_config),
    model_manager: ModelManager = Depends(get_model_manager),
):
    logger.info("Attempting to acquire miner lock...")
    async with miner_lock:
        logger.info("Miner lock acquired, processing challenge...")
        try:
            challenge_data = await request.json()
            challenge_id = challenge_data.get("challenge_id")
            video_url = challenge_data.get("video_url")
            
            logger.info(f"Received challenge data: {json.dumps(challenge_data, indent=2)}")
            
            if not video_url:
                raise HTTPException(status_code=400, detail="No video URL provided")
            
            logger.info(f"Processing challenge {challenge_id} with video {video_url}")
            
            video_path = await download_video(video_url)
            
            try:
                tracking_data = await process_soccer_video(
                    video_path,
                    model_manager
                )
                
                response = {
                    "challenge_id": challenge_id,
                    "frames": tracking_data["frames"],
                    "processing_time": tracking_data["processing_time"]
                }
                
                logger.info(f"""
                    ==== Response ====
                    📌 Challenge ID: {response['challenge_id']}
                    📌 Processing Time: {response['processing_time']:.2f} seconds
                    📌 Frames Processed: {len(response['frames'])}
                    ==================
                """)

                # logger.info(f"Completed challenge {challenge_id} in {tracking_data['processing_time']:.2f} seconds")
                return response
                
            finally:
                try:
                    os.unlink(video_path)
                except:
                    pass
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error processing soccer challenge: {str(e)}")
            logger.exception("Full error traceback:")
            raise HTTPException(status_code=500, detail=f"Challenge processing error: {str(e)}")
        finally:
            logger.info("Releasing miner lock...")

# Create router with dependencies
router = APIRouter()
router.add_api_route(
    "/challenge",
    process_challenge,
    tags=["soccer"],
    dependencies=[Depends(blacklist_low_stake), Depends(verify_request)],
    methods=["POST"],
)
EOL

echo "File soccer.py telah diperbarui."

rm -f score-vision/miner/utils/video_downloader.py
cat > score-vision/miner/utils/video_downloader.py <<EOL
import tempfile
from pathlib import Path
import httpx
from fastapi import HTTPException
from tenacity import retry, stop_after_attempt, wait_exponential
from loguru import logger

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def download_video(url: str) -> Path:
    """
    Download video with retries and proper redirect handling.
    
    Args:
        url: URL of the video to download
        
    Returns:
        Path: Path to the downloaded video file
        
    Raises:
        HTTPException: If download fails
    """
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            # First request to get the redirect
            response = await client.get(url)
            
            if "drive.google.com" in url:
                # For Google Drive, we need to handle the download URL specially
                if "drive.usercontent.google.com" in response.url.path:
                    download_url = str(response.url)
                else:
                    # If we got redirected to the Google Drive UI, construct the direct download URL
                    file_id = url.split("id=")[1].split("&")[0]
                    download_url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download"
                
                # Make the actual download request
                response = await client.get(download_url)
            
            response.raise_for_status()
            
            # Create temp file with .mp4 extension
            temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
            temp_file.write(response.content)
            temp_file.close()
            
            logger.info(f"Video downloaded successfully to {temp_file.name}")
            return Path(temp_file.name)
            
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error downloading video: {str(e)}")
        logger.error(f"Response status code: {e.response.status_code}")
        logger.error(f"Response headers: {e.response.headers}")
        logger.error(f"vidionya bego: {e.detail}")
        raise HTTPException(status_code=500, detail=f"Failed to download video: {str(e)}")
    except Exception as e:
        logger.error(f"Error downloading video: {str(e)}")
        logger.error(f"vidionya bego: {e.detail}")
        raise HTTPException(status_code=500, detail=f"Failed to download video: {str(e)}") 
EOL

echo "File video_downloader.py telah diperbarui."

# Loop untuk menyalin folder sebanyak yang diminta

for ((i=2; i<=CLONE_COUNT; i++))
do
    cp -r score-vision "score-vision$i"
    echo "Berhasil menyalin score-vision ke score-vision$i"
done

# Loop untuk setup masing-masing folder
for ((i=1; i<=CLONE_COUNT; i++))
do
    # TARGET_DIR="score-vision$i"
    if [ "$i" -eq 1 ]; then
        TARGET_DIR="score-vision"
    else
        TARGET_DIR="score-vision$i"
    fi
    echo "Masuk ke direktori $TARGET_DIR dan menjalankan setup..."
    
    cd "$TARGET_DIR" || { echo "Gagal masuk ke $TARGET_DIR"; exit 1; }

    # Jalankan proses instalasi di setiap folder
    chmod +x bootstrap.sh
    ./bootstrap.sh
    # ==== FIX STARTS HERE ====
    # Ensure uv is in PATH
    source ~/.bashrc                  # Load updated PATH
    export PATH="$HOME/.local/bin:$PATH"  # Explicitly add uv's path
    # ==== FIX ENDS HERE ====
    uv venv
    source .venv/bin/activate
    uv pip install -e ".[miner]"

    # Buat file miner/.env dengan konfigurasi yang sesuai
    cat > miner/.env <<EOL
# Subnet Configuration (Mainnet 44)
NETUID=44
SUBTENSOR_NETWORK=finney
WALLET_NAME=default
HOTKEY_NAME=$i
MIN_STAKE_THRESHOLD=1000
SUBTENSOR_ADDRESS=wss://entrypoint-finney.opentensor.ai:443
DEVICE=cuda
EOL

    echo "File miner/.env telah dibuat dengan HOTKEY_NAME=$i"

    uv pip install "git+https://github.com/rayonlabs/fiber.git@2.1.0#egg=fiber[full]"

    # Kembali ke direktori utama sebelum lanjut ke clone berikutnya
    cd ..
done

echo "Selesai! Total $CLONE_COUNT folder telah di-clone dan diinstal."
