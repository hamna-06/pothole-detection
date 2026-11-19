# Real-Time Pothole Detection (YOLO11s)

Potholes damage vehicles and cause accidents, and checking roads by hand is slow. This project detects potholes
automatically in road images, recorded videos and a live camera feed, and shows the result on screen while it runs.

**Model:** YOLO11s (Ultralytics), fine-tuned from COCO weights on a public pothole dataset.
**Results (held-out test set):** precision 0.772, recall 0.747, F1 0.759, mAP@0.5 0.806.
**Speed:** about 7 ms per image on a Tesla T4 GPU.

## Features
- Image, video and live-camera detection
- Bounding boxes with confidence scores
- Unique pothole counter using ByteTrack
- FPS display and Road clear / Warning / Danger status
- Download of annotated images and videos

## Run
pip install -r requirements.txt
streamlit run app.py

## Files
- app.py: Streamlit application
- best.pt: trained weights
- notebook/: Kaggle notebook (EDA, cleaning, training, evaluation)

Made by: [Name 1], [Name 2], [Name 3], [Name 4]
Jinnah University for Women, Department of Computer Science and Software Engineering
Course: Artificial Neural Networks
