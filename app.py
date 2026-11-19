import os, time, tempfile
import cv2, numpy as np, pandas as pd, streamlit as st
from ultralytics import YOLO

# ------------------------------------------------------------------ CONFIG
PROJECT_TITLE = "Pothole Detection System"
AUTHOR = "Areesha Sajid | Hamna Raees | Khizra Anwar  | Nabiha Arshad | Artificial Neural Networks  |  Final Project"
# Replace with YOUR held-out TEST results from the Kaggle notebook (Cell 11)
METRICS = {"Precision": "0.772", "Recall": "0.747", "mAP@0.5": "0.806", "mAP@0.5:0.95": "0.446"}

st.set_page_config(page_title=PROJECT_TITLE, page_icon="🚧", layout="wide")

# ------------------------------------------------------------------ STYLE
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Poppins', sans-serif; }

/* ---------- animated road-tech background ---------- */
.stApp {
  background:
    radial-gradient(1100px 600px at 88% -8%, rgba(255,193,7,.26), transparent 60%),
    radial-gradient(900px 650px at -8% 105%, rgba(0,200,255,.20), transparent 60%),
    linear-gradient(rgba(255,255,255,.04) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,.04) 1px, transparent 1px),
    linear-gradient(135deg, #04060b 0%, #0a1120 50%, #111c32 100%);
  background-size: auto, auto, 52px 52px, 52px 52px, auto;
  background-attachment: fixed;
}
.stApp::before {
  content: ""; position: fixed; width: 520px; height: 520px; border-radius: 50%;
  top: 8%; left: 55%; background: radial-gradient(circle, rgba(255,160,0,.16), transparent 65%);
  filter: blur(30px); animation: floaty 14s ease-in-out infinite alternate; pointer-events: none; z-index: 0;
}
@keyframes floaty { from { transform: translate(0,0); } to { transform: translate(-220px,160px); } }

/* ---------- readable glass panel (content never fades into the background) ---------- */
header[data-testid="stHeader"] { background: transparent; }
.block-container {
  max-width: 1280px; margin-top: 1rem; padding: 2rem 2.6rem 2.5rem !important;
  background: rgba(7,11,20,.86); backdrop-filter: blur(16px);
  border: 1px solid rgba(255,255,255,.09); border-radius: 26px;
  box-shadow: 0 25px 80px rgba(0,0,0,.55);
}
.stApp, .stMarkdown, p, li { color: #E8EDF5; }
[data-testid="stWidgetLabel"] p { color: #D5DEEB !important; font-weight: 500; }

/* ---------- hero ---------- */
.hazard { height: 9px; border-radius: 6px; margin-bottom: 22px;
  background: repeating-linear-gradient(45deg, #FFC107 0 16px, #151515 16px 32px); }
.hero h1 { font-size: 3rem; font-weight: 800; margin: 6px 0 4px; line-height: 1.1; color: #fff; }
.hero h1 span { background: linear-gradient(90deg, #FFC107, #FF8F00); -webkit-background-clip: text;
  -webkit-text-fill-color: transparent; }
.hero p { font-size: 1.05rem; color: #B8C4D6; max-width: 820px; margin: 0 0 14px; }
.badge { display: inline-block; padding: 5px 14px; border-radius: 999px; font-size: .75rem; letter-spacing: .12em;
  font-weight: 600; color: #FFC107; border: 1px solid rgba(255,193,7,.5); background: rgba(255,193,7,.08); }
.chip { display: inline-block; margin: 4px 8px 4px 0; padding: 6px 14px; border-radius: 12px; font-size: .82rem;
  background: rgba(255,255,255,.07); border: 1px solid rgba(255,255,255,.12); color: #E2E8F0; }

/* ---------- tabs ---------- */
.stTabs [data-baseweb="tab-list"] { gap: 10px; border: none; }
.stTabs button[data-baseweb="tab"] { background: rgba(255,255,255,.06); border-radius: 12px; padding: 10px 22px;
  color: #CBD5E1; font-weight: 600; border: 1px solid rgba(255,255,255,.08); }
.stTabs button[aria-selected="true"] { background: linear-gradient(135deg, #FFC107, #FF8F00) !important;
  color: #111 !important; border: none; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] { background: rgba(5,8,15,.95); border-right: 1px solid rgba(255,193,7,.28); }
section[data-testid="stSidebar"] h2 { color: #FFC107; }

/* ---------- widgets ---------- */
[data-testid="stFileUploader"] section { background: rgba(255,255,255,.04); border: 2px dashed rgba(255,193,7,.55);
  border-radius: 16px; }
[data-testid="stFileUploader"] section:hover { background: rgba(255,193,7,.07); }
.stButton > button, .stDownloadButton > button { background: linear-gradient(135deg, #FFC107, #FF8F00); color: #111;
  font-weight: 700; border: none; border-radius: 12px; padding: .6rem 1.5rem; transition: all .2s; }
.stButton > button:hover, .stDownloadButton > button:hover { transform: translateY(-2px);
  box-shadow: 0 8px 22px rgba(255,160,0,.4); color: #000; }
[data-testid="stImage"] img { border-radius: 16px; border: 1px solid rgba(255,255,255,.14);
  box-shadow: 0 12px 34px rgba(0,0,0,.5); }
[data-testid="stImageCaption"] { color: #9FB0C7; }
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

/* ---------- stat cards & alerts ---------- */
.stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 14px; margin: 12px 0; }
.stat { padding: 16px 18px; border-radius: 16px; background: linear-gradient(145deg, rgba(255,255,255,.08), rgba(255,255,255,.02));
  border: 1px solid rgba(255,255,255,.12); border-left: 4px solid #FFC107; }
.stat-v { font-size: 1.9rem; font-weight: 800; color: #fff; line-height: 1.1; }
.stat-l { font-size: .78rem; text-transform: uppercase; letter-spacing: .09em; color: #9FB0C7; margin-top: 4px; }
.alert { padding: 14px 20px; border-radius: 14px; font-weight: 700; font-size: 1.05rem; margin: 8px 0; }
.ok { background: rgba(34,197,94,.14); border: 1px solid #22c55e; color: #86efac; }
.warn { background: rgba(255,152,0,.15); border: 1px solid #ff9800; color: #ffcc80; }
.danger { background: rgba(239,68,68,.20); border: 1px solid #ef4444; color: #fecaca; animation: pulse 1s infinite; }
@keyframes pulse { 0%,100% { box-shadow: 0 0 0 0 rgba(239,68,68,.55); } 50% { box-shadow: 0 0 0 14px rgba(239,68,68,0); } }
.card { padding: 20px 24px; border-radius: 18px; background: rgba(255,255,255,.05); border: 1px solid rgba(255,255,255,.1); height: 100%; }
.card h4 { color: #FFC107; margin-top: 0; }
.footer { text-align: center; color: #7F8FA6; font-size: .85rem; margin-top: 28px; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------ HERO
st.markdown(f"""
<div class="hero">
  <div class="hazard"></div>
  <span class="badge">DEEP LEARNING  ·  COMPUTER VISION  ·  ROAD SAFETY</span>
  <h1>Pothole <span>Detection</span> System</h1>
  <p>Real-time pothole detection powered by a fine-tuned YOLO11 neural network. Upload road images or videos,
  or connect a live camera to detect, count and flag dangerous potholes instantly.</p>
  <span class="chip">🧠 YOLO11s</span><span class="chip">⚡ Real-time</span>
  <span class="chip">🎯 Object tracking</span><span class="chip">🚨 Danger alerts</span>
</div>
""", unsafe_allow_html=True)
st.write("")

# ------------------------------------------------------------------ SIDEBAR
with st.sidebar:
    st.markdown("## ⚙️ Control Panel")
    model_path = st.text_input("Model weights", "best.pt")
    conf = st.slider("Confidence threshold", 0.05, 0.95, 0.35, 0.05,
                     help="Higher = fewer false alarms, lower = fewer missed potholes.")
    iou_thr = st.slider("NMS IoU threshold", 0.30, 0.90, 0.50, 0.05)
    imgsz = st.selectbox("Inference size (px)", [416, 640, 800, 960], index=1,
                         help="Smaller = faster, larger = better for small potholes.")
    danger_area = st.slider("Danger zone (% of frame)", 0.1, 10.0, 1.0, 0.1,
                            help="A larger box means the pothole is closer to the camera.")
    skip = st.slider("Process every Nth frame", 1, 5, 1, help="Increase on slow laptops for smoother playback.")
    st.markdown("---")
    st.caption("💡 Tip: a bigger bounding box means the pothole is closer to the vehicle.")

if not os.path.exists(model_path):
    st.error(f"Model file '{model_path}' not found. Place best.pt next to app.py.")
    st.stop()

@st.cache_resource
def load_model(p): return YOLO(p)

# ------------------------------------------------------------------ HELPERS
FONT = cv2.FONT_HERSHEY_SIMPLEX

def coverage(box, shape):
    x1, y1, x2, y2 = box
    return (x2 - x1) * (y2 - y1) / (shape[0] * shape[1]) * 100

def level_of(cov_list):
    if not cov_list: return None
    return "DANGER" if max(cov_list) >= danger_area else "WARNING"

def risk_label(c):
    return "🔴 High" if c >= danger_area else ("🟠 Medium" if c >= danger_area / 3 else "🟢 Low")

def stats_html(items):
    return '<div class="stats">' + "".join(
        f'<div class="stat"><div class="stat-v">{v}</div><div class="stat-l">{l}</div></div>' for l, v in items) + "</div>"

def alert_html(level):
    if level == "DANGER": return '<div class="alert danger">🚨 DANGER: Pothole very close. Slow down!</div>'
    if level == "WARNING": return '<div class="alert warn">⚠️ WARNING: Pothole ahead</div>'
    return '<div class="alert ok">✅ Road clear: no pothole detected</div>'

def hud(img, n, total, fps, level):
    h, w = img.shape[:2]
    ov = img.copy(); cv2.rectangle(ov, (0, 0), (w, 44), (15, 15, 15), -1)
    img = cv2.addWeighted(ov, .65, img, .35, 0)
    cv2.putText(img, f"FPS {fps:.1f}   IN FRAME {n}   UNIQUE TOTAL {total}", (12, 29), FONT, .7, (0, 200, 255), 2, cv2.LINE_AA)
    if level:
        col = (40, 40, 230) if level == "DANGER" else (0, 150, 255)
        cv2.rectangle(img, (0, 48), (w, 92), col, -1)
        msg = "DANGER - POTHOLE VERY CLOSE" if level == "DANGER" else "WARNING - POTHOLE AHEAD"
        cv2.putText(img, msg, (12, 80), FONT, .95, (255, 255, 255), 3, cv2.LINE_AA)
    return img

def run_stream(cap, save_path=None):
    model = YOLO(model_path)                      # fresh model = fresh tracker for every run
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    alert_ph, stats_ph, frame_ph = st.empty(), st.empty(), st.empty()
    bar = st.progress(0.0) if (save_path and total_frames > 0) else None
    seen, writer, idx, fps, prev, peak, done = set(), None, 0, 0.0, time.time(), 0, 0
    try:
        while cap.isOpened():
            if not cap.grab(): break
            idx += 1
            if skip > 1 and idx % skip: continue
            ok, frame = cap.retrieve()
            if not ok: break
            r = model.track(frame, persist=True, conf=conf, iou=iou_thr, imgsz=imgsz, verbose=False)[0]
            if r.boxes.id is not None: seen.update(r.boxes.id.int().cpu().tolist())
            covs = [coverage(b, frame.shape) for b in r.boxes.xyxy.cpu().numpy()]
            level = level_of(covs); n = len(r.boxes); peak = max(peak, n); done += 1
            now = time.time(); fps = 0.9 * fps + 0.1 / max(now - prev, 1e-6) if fps else 1 / max(now - prev, 1e-6); prev = now
            out = hud(r.plot(), n, len(seen), fps, level)
            frame_ph.image(out, channels="BGR")
            alert_ph.markdown(alert_html(level), unsafe_allow_html=True)
            stats_ph.markdown(stats_html([("In frame", n), ("Unique potholes", len(seen)),
                                          ("FPS", f"{fps:.1f}"), ("Frames processed", done)]), unsafe_allow_html=True)
            if bar is not None: bar.progress(min(idx / total_frames, 1.0))
            if save_path:
                if writer is None:
                    h, w = out.shape[:2]
                    writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), max(src_fps / skip, 1), (w, h))
                writer.write(out)
    finally:
        cap.release()
        if writer: writer.release()
    return len(seen), peak, done

# ------------------------------------------------------------------ TABS
tab_img, tab_vid, tab_cam, tab_about = st.tabs(["🖼️  Image Detection", "🎞️  Video Detection", "📷  Live Camera", "ℹ️  About Project"])

# ---- IMAGES
with tab_img:
    st.markdown("### Upload road images")
    files = st.file_uploader("Drag and drop one or more images", type=["jpg", "jpeg", "png", "bmp", "webp"],
                             accept_multiple_files=True, key="imgs")
    if files:
        model = load_model(model_path)
        results, total_p = [], 0
        for f in files:
            arr = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
            t = time.time(); r = model.predict(arr, conf=conf, iou=iou_thr, imgsz=imgsz, verbose=False)[0]
            results.append((f.name, arr, r, (time.time() - t) * 1000)); total_p += len(r.boxes)
        st.markdown(stats_html([("Images analysed", len(files)), ("Potholes found", total_p),
                                ("Avg. time / image", f"{np.mean([x[3] for x in results]):.0f} ms")]), unsafe_allow_html=True)
        for name, arr, r, ms in results:
            st.markdown(f"#### 📄 {name}")
            c1, c2 = st.columns(2)
            c1.image(arr, channels="BGR", caption="Original image")
            ann = r.plot()
            c2.image(ann, channels="BGR", caption=f"Detection result: {len(r.boxes)} pothole(s) in {ms:.0f} ms")
            covs = [coverage(b, arr.shape) for b in r.boxes.xyxy.cpu().numpy()]
            st.markdown(alert_html(level_of(covs)), unsafe_allow_html=True)
            if len(r.boxes):
                st.dataframe(pd.DataFrame({"#": range(1, len(covs) + 1),
                                           "Confidence": r.boxes.conf.cpu().numpy().round(3),
                                           "Frame coverage (%)": np.round(covs, 2),
                                           "Risk level": [risk_label(c) for c in covs]}),
                             hide_index=True)
            st.download_button("⬇ Download result", cv2.imencode(".jpg", ann)[1].tobytes(),
                               f"detected_{name}", "image/jpeg", key=f"dl_{name}")
            st.markdown("---")
    else:
        st.info("Upload one or more road images to start detection.")

# ---- VIDEO
with tab_vid:
    st.markdown("### Upload a road video")
    vf = st.file_uploader("Supported formats: MP4, AVI, MOV, MKV", type=["mp4", "avi", "mov", "mkv"], key="vid")
    if vf:
        st.video(vf)
        if st.button("▶ Start detection", key="vstart"):
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4"); tmp.write(vf.getvalue()); tmp.close()
            out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            t0 = time.time()
            uniq, peak, frames = run_stream(cv2.VideoCapture(tmp.name), save_path=out_path)
            el = time.time() - t0
            st.success("Processing complete!")
            st.markdown(stats_html([("Unique potholes", uniq), ("Max in one frame", peak),
                                    ("Frames processed", frames), ("Average FPS", f"{frames / max(el, 1e-6):.1f}")]),
                        unsafe_allow_html=True)
            st.download_button("⬇ Download annotated video", open(out_path, "rb").read(), "pothole_result.mp4", "video/mp4")
    else:
        st.info("Upload a road video, then press **Start detection**.")

# ---- LIVE CAMERA
with tab_cam:
    st.markdown("### Live camera detection")
    src = st.text_input("Camera source", "0",
                        help="0 = laptop webcam. For a phone camera, use an IP-camera app URL, e.g. http://192.168.1.5:8080/video")
    b1, b2, _ = st.columns([1, 1, 4])
    start = b1.button("▶ Start camera", key="cstart"); b2.button("⏹ Stop", key="cstop")
    if start:
        cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
        if not cap.isOpened(): st.error("Could not open the camera. Close other apps using it, or try source 1.")
        else: run_stream(cap)
    else:
        st.info("Press **Start camera** to begin live detection. Press **Stop** to end the session.")

# ---- ABOUT
with tab_about:
    st.markdown("### About this project")
    a, b = st.columns(2)
    a.markdown("""<div class="card"><h4>🎯 Problem</h4>
    Potholes cause accidents and vehicle damage, and manual road surveys are slow and costly.
    This system detects potholes automatically from images, video and live camera feeds so that drivers and
    road authorities can react earlier.</div>""", unsafe_allow_html=True)
    b.markdown("""<div class="card"><h4>🧠 Method</h4>
    A YOLO11s convolutional neural network was fine-tuned on a public pothole dataset after cleaning,
    de-duplication, annotation verification and augmentation. ByteTrack assigns IDs so each pothole is counted once.</div>""",
               unsafe_allow_html=True)
    st.write("")
    st.markdown("#### 📊 Model performance (held-out test set)")
    st.markdown(stats_html(list(METRICS.items())), unsafe_allow_html=True)
    st.markdown("""<div class="card"><h4>⚠️ Limitations</h4>
    Small or distant potholes, heavy shadows, water-filled potholes and manhole covers can cause misses or false alarms.
    The unique-pothole counter relies on tracking and may over-count if an object ID switches.</div>""", unsafe_allow_html=True)

st.markdown(f'<div class="footer">{AUTHOR}<br>Built with YOLO11 · Ultralytics · OpenCV · Streamlit</div>', unsafe_allow_html=True)
