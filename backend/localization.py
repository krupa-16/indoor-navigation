
import base64
from pathlib import Path
from io import BytesIO
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from fastapi import APIRouter, File, HTTPException, UploadFile

router = APIRouter()
BASE_DIR = Path(__file__).parent
LANDMARK_DIR = BASE_DIR / "landmarks"
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_IMAGE_SIZE = 8 * 1024 * 1024
MIN_INLIERS = 8
MIN_INLIER_RATIO = 0.25
orb = cv2.ORB_create(nfeatures=1500)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

def decode_image(raw):
    try:
        image = Image.open(BytesIO(raw)).convert("RGB")
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(400, "Invalid image file.")

def read_upload(file, raw):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(400, "Upload a JPG, PNG, or WebP image.")
    if len(raw) > MAX_IMAGE_SIZE:
        raise HTTPException(413, "Image must be 8 MB or smaller.")
    return decode_image(raw)

@router.post("/api/localize")
async def localize(file: UploadFile = File(...)):
    raw = await file.read(MAX_IMAGE_SIZE + 1)
    query_gray = read_upload(file, raw)
    query_keypoints, query_descriptors = orb.detectAndCompute(query_gray, None)
    if query_descriptors is None or len(query_keypoints) < 8:
        raise HTTPException(422, "Not enough visual features in the query image.")
    if not LANDMARK_DIR.exists():
        raise HTTPException(404, "Create backend/landmarks and add reference images.")
    reference_files = sorted(
        p for p in LANDMARK_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if not reference_files:
        raise HTTPException(404, "No landmark images found in backend/landmarks.")
    results = []
    for path in reference_files:
        reference_gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if reference_gray is None:
            continue
        ref_keypoints, ref_descriptors = orb.detectAndCompute(reference_gray, None)
        if ref_descriptors is None or len(ref_keypoints) < 4:
            continue
        pairs = matcher.knnMatch(query_descriptors, ref_descriptors, k=2)
        good_matches = [
            pair[0] for pair in pairs
            if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance
        ]
        inlier_matches = []
        if len(good_matches) >= 4:
            query_points = np.float32(
                [query_keypoints[m.queryIdx].pt for m in good_matches]
            ).reshape(-1, 1, 2)
            reference_points = np.float32(
                [ref_keypoints[m.trainIdx].pt for m in good_matches]
            ).reshape(-1, 1, 2)
            try:
                homography, mask = cv2.findHomography(
                    query_points, reference_points, cv2.RANSAC, 5.0
                )
                if homography is not None and mask is not None:
                    inlier_matches = [
                        match for match, keep in zip(good_matches, mask.ravel())
                        if keep
                    ]
            except cv2.error:
                inlier_matches = []
        inlier_count = len(inlier_matches)
        inlier_ratio = inlier_count / len(good_matches) if good_matches else 0.0
        results.append({
            "path": path,
            "gray": reference_gray,
            "keypoints": ref_keypoints,
            "good_matches": good_matches,
            "inlier_matches": inlier_matches,
            "raw_count": len(good_matches),
            "inlier_count": inlier_count,
            "inlier_ratio": inlier_ratio,
        })
    if not results:
        raise HTTPException(422, "Could not extract enough features from the reference images.")
    results.sort(
        key=lambda item: (item["inlier_count"], item["inlier_ratio"], item["raw_count"]),
        reverse=True,
    )
    best = results[0]
    localized = (
        best["inlier_count"] >= MIN_INLIERS
        and best["inlier_ratio"] >= MIN_INLIER_RATIO
    )
    top_results = []
    for item in results[:3]:
        match_image = None
        if item["inlier_matches"]:
            visual_matches = item["inlier_matches"][:60]
            drawn = cv2.drawMatches(
                query_gray, query_keypoints,
                item["gray"], item["keypoints"],
                visual_matches, None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            )
            ok, encoded = cv2.imencode(".jpg", drawn)
            if ok:
                match_image = "data:image/jpeg;base64," + base64.b64encode(
                    encoded
                ).decode("ascii")
        top_results.append({
            "landmark": item["path"].stem.replace("_", " ").title(),
            "good_matches": item["raw_count"],
            "verified_matches": item["inlier_count"],
            "inlier_ratio": round(item["inlier_ratio"], 3),
            "match_image": match_image,
        })
    return {
        "localized": localized,
        "best_match": best["path"].stem.replace("_", " ").title() if localized else None,
        "good_matches": best["raw_count"],
        "verified_matches": best["inlier_count"],
        "inlier_ratio": round(best["inlier_ratio"], 3),
        "message": (
            "Landmark matched with geometric verification."
            if localized
            else "No sufficiently reliable landmark match. Try another image or add better reference images."
        ),
        "matches": top_results,
    }
