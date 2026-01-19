# =========================================================
# THUMBNAIL ENRICHMENT PIPELINE (CON LOGS)
# =========================================================

import requests
import pandas as pd
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import cv2
import threading
import time

# =========================================================
# CONFIGURACIÓN
# =========================================================
MAX_WORKERS = 6        
TIMEOUT = 15
LOG_EVERY = 50         # Log cada N imágenes

HEADERS = {
    "User-Agent": "Mozilla/5.0 (thumbnail-analyzer/1.0)"
}

# =========================================================
# CONTADORES GLOBALES (THREAD-SAFE)
# =========================================================
counter_lock = threading.Lock()
processed_count = 0
start_time = time.time()

# =========================================================
# FUNCIÓN: PROCESAR UNA FILA
# =========================================================
def process_thumbnail_row(row):
    global processed_count

    try:
        video_id = row["video_id"]
        url = row["thumbnail_link"]

        if pd.isna(url):
            raise ValueError("thumbnail_link vacío")

        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()

        img = Image.open(BytesIO(r.content)).convert("RGB")
        img_np = np.array(img)

        height, width = img_np.shape[:2]
        aspect_ratio = round(width / height, 3)

        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        brightness = float(gray.mean())
        contrast = float(gray.std())
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        saturation = float(hsv[:, :, 1].mean())

        with counter_lock:
            processed_count += 1
            if processed_count % LOG_EVERY == 0:
                elapsed = time.time() - start_time
                print(f"✅ Procesadas {processed_count} imágenes | {elapsed:.1f}s")

        return {
            "video_id": video_id,
            "thumb_width": width,
            "thumb_height": height,
            "aspect_ratio": aspect_ratio,
            "brightness": brightness,
            "contrast": contrast,
            "sharpness": sharpness,
            "saturation": saturation,
            "thumb_ok": True,
            "thumb_error": None
        }

    except Exception as e:
        with counter_lock:
            processed_count += 1
            print(f"❌ Error en video_id={row.get('video_id')} | {str(e)}")

        return {
            "video_id": row.get("video_id"),
            "thumb_width": None,
            "thumb_height": None,
            "aspect_ratio": None,
            "brightness": None,
            "contrast": None,
            "sharpness": None,
            "saturation": None,
            "thumb_ok": False,
            "thumb_error": str(e)
        }

# =========================================================
# FUNCIÓN: ENRIQUECER DATAFRAME EN PARALELO
# =========================================================
def enrich_df_with_thumbnails(df: pd.DataFrame) -> pd.DataFrame:
    print(f"🚀 Iniciando procesamiento de {len(df)} thumbnails")
    results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(process_thumbnail_row, row)
            for _, row in df.iterrows()
        ]

        for future in as_completed(futures):
            results.append(future.result())

    print(f"🎯 Procesamiento terminado. Total procesadas: {processed_count}")
    return df.merge(pd.DataFrame(results), on="video_id", how="left")

# =========================================================
# EJECUCIÓN
# =========================================================
if __name__ == "__main__":

    print("📂 Cargando dataset...")
    df = pd.read_csv("datasets/GBvideos.csv")

    required_cols = {"video_id", "thumbnail_link"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"El dataset debe contener las columnas: {required_cols}")

    df_enriched = enrich_df_with_thumbnails(df)

    df_enriched.to_csv("dataset_enriched.csv", index=False)

    total_time = time.time() - start_time
    print(f"🏁 Proceso terminado en {total_time:.1f} segundos")
    print("📄 Archivo generado: dataset_enriched.csv")
