"""
live.py — Intrusion Detection GUI
  • Register User : MTCNN scan → ArcFace embedding → SQLite
  • Live Detect   : MTCNN scan every 2 s → match DB → name or INTRUDER
"""
import tkinter as tk
from tkinter import messagebox, simpledialog
import cv2
import numpy as np
import onnxruntime as ort
import sqlite3
import time
import threading
from PIL import Image, ImageTk

# ── Config ─────────────────────────────────────────────────────────────────
ONNX_MODEL       = "artifacts/arcface_model.onnx"
DB_FILE          = "users.db"
IMG_SIZE         = 112
THRESHOLD        = 0.50
DETECT_INTERVAL  = 2.0    # seconds between recognition cycles
PREVIEW_INTERVAL = 1.0    # box refresh in register mode
FRAME_MS         = 30     # ~33 fps display

# ── MTCNN (facenet-pytorch) ────────────────────────────────────────────────
try:
    from facenet_pytorch import MTCNN as _MTCNN
    _detector = _MTCNN(keep_all=True, device="cpu")
except ImportError:
    raise SystemExit("facenet-pytorch not found — run:  pip install facenet-pytorch")

def detect_faces(bgr):
    """Returns list of {'box':[x,y,w,h], 'confidence':float}."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    boxes, probs = _detector.detect(rgb)
    if boxes is None:
        return []
    out = []
    for box, prob in zip(boxes, probs):
        if prob is None or prob < 0.90:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        out.append({"box": [x1, y1, x2 - x1, y2 - y1], "confidence": float(prob)})
    return out

# ── ArcFace (CPU ONNX) ─────────────────────────────────────────────────────
_sess  = ort.InferenceSession(ONNX_MODEL, providers=["CPUExecutionProvider"])
_iname = _sess.get_inputs()[0].name

def get_embedding(bgr_crop):
    img = cv2.resize(bgr_crop, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = (img.astype(np.float32) / 255.0 - 0.5) / 0.5
    img = np.transpose(img, (2, 0, 1))[np.newaxis]
    emb = _sess.run(None, {_iname: img})[0][0]
    return emb / np.linalg.norm(emb)

def cosine(a, b):
    return float(np.dot(a, b))

# ── SQLite ─────────────────────────────────────────────────────────────────
def init_db():
    with sqlite3.connect(DB_FILE) as c:
        c.execute("""CREATE TABLE IF NOT EXISTS users (
            id   INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT    NOT NULL,
            emb  BLOB    NOT NULL)""")

def save_user(name, emb):
    with sqlite3.connect(DB_FILE) as c:
        c.execute("INSERT INTO users (name, emb) VALUES (?,?)",
                  (name, emb.astype(np.float32).tobytes()))

def load_users():
    with sqlite3.connect(DB_FILE) as c:
        rows = c.execute("SELECT name, emb FROM users").fetchall()
    return [(n, np.frombuffer(e, dtype=np.float32)) for n, e in rows]

def match_db(emb):
    users = load_users()
    if not users:
        return None, 0.0
    sims = [(n, cosine(emb, e)) for n, e in users]
    name, sim = max(sims, key=lambda x: x[1])
    return (name, sim) if sim >= THRESHOLD else (None, sim)

# ── Colours ────────────────────────────────────────────────────────────────
BG     = "#0d1117"
PANEL  = "#161b22"
ACCENT = "#58a6ff"
RED    = "#f85149"
GREEN  = "#3fb950"
ORANGE = "#d29922"
TEXT   = "#e6edf3"
MUTED  = "#8b949e"

def hex2bgr(h):
    h = h.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))

# ── App ────────────────────────────────────────────────────────────────────
class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Live — Intrusion Detection")
        self.root.configure(bg=BG)
        self.root.geometry("720x540")
        self.root.minsize(520, 400)
        self.root.resizable(True, True)

        self._lock         = threading.Lock()
        self._boxes: list  = []
        self._banner       = ""
        self._banner_col   = TEXT

        self.cap           = None
        self._frame_lock   = threading.Lock()
        self._latest_frame = None
        self.mode          = None
        self._running      = False
        self._reg_name     = ""

        init_db()
        self._build_ui()
        self.root.bind("<space>", lambda _: self._capture_register())
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI ─────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # header row
        hdr = tk.Frame(self.root, bg=PANEL, height=42)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="Intrusion Detection  •  Live",
                 font=("Segoe UI", 12, "bold"),
                 bg=PANEL, fg=ACCENT).pack(side="left", padx=14, pady=8)

        # button row
        bar = tk.Frame(self.root, bg=BG)
        bar.pack(fill="x", padx=10, pady=6)
        for txt, col, cmd in [
            ("Register User", ACCENT, self._on_register),
            ("Live Detect",   GREEN,  self._on_live),
            ("Stop",          RED,    self._stop),
        ]:
            tk.Button(bar, text=txt, font=("Segoe UI", 10, "bold"),
                      bg=PANEL, fg=col,
                      activebackground=col, activeforeground=BG,
                      relief="flat", bd=0, padx=14, pady=6,
                      cursor="hand2", command=cmd
                      ).pack(side="left", padx=6)

        # video — fills all remaining space
        self.canvas = tk.Label(self.root, bg="#000")
        self.canvas.pack(fill="both", expand=True, padx=10, pady=4)

        # status + hint at bottom
        bottom = tk.Frame(self.root, bg=BG)
        bottom.pack(fill="x", padx=10, pady=(0, 6))

        self._status = tk.Label(bottom, text="Ready",
                                font=("Segoe UI", 11, "bold"),
                                bg=BG, fg=TEXT, anchor="w")
        self._status.pack(side="left")

        self._hint = tk.Label(bottom, text="",
                              font=("Segoe UI", 9),
                              bg=BG, fg=MUTED, anchor="e")
        self._hint.pack(side="right")

    # ── Handlers ───────────────────────────────────────────────────────────
    def _on_register(self):
        name = simpledialog.askstring("Register", "Enter full name:", parent=self.root)
        if not name or not name.strip():
            return
        self._reg_name = name.strip()
        self.mode      = "register"
        self._set_status(f"Registering: {self._reg_name}", ORANGE)
        self._hint.config(text="Press  SPACE  to capture")
        self._open_camera()
        threading.Thread(target=self._worker, daemon=True).start()

    def _on_live(self):
        self.mode = "detect"
        self._set_status("Live Detection Active", GREEN)
        self._hint.config(text=f"Cycle every {DETECT_INTERVAL:.0f} s")
        self._open_camera()
        threading.Thread(target=self._worker, daemon=True).start()

    def _stop(self):
        self._running = False
        self.mode     = None
        if self.cap:
            self.cap.release()
            self.cap = None
        with self._lock:
            self._boxes      = []
            self._banner     = ""
            self._banner_col = TEXT
        self._set_status("Stopped", MUTED)
        self._hint.config(text="")
        self._show(np.zeros((360, 480, 3), np.uint8))

    def _on_close(self):
        self._stop()
        self.root.destroy()

    # ── Camera / display ───────────────────────────────────────────────────
    def _open_camera(self):
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Camera", "Cannot open webcam.")
            return
        self._running = True
        threading.Thread(target=self._frame_reader, daemon=True).start()
        self._display_loop()

    def _frame_reader(self):
        while self._running and self.cap:
            ret, frame = self.cap.read()
            if ret:
                with self._frame_lock:
                    self._latest_frame = cv2.flip(frame, 1)
            time.sleep(0.01)

    def _get_frame(self):
        with self._frame_lock:
            f = self._latest_frame
        return f.copy() if f is not None else None

    def _display_loop(self):
        if not self._running or self.cap is None:
            return
        frame = self._get_frame()
        if frame is not None:
            with self._lock:
                boxes  = list(self._boxes)
                banner = self._banner
                bcol   = self._banner_col
            out = frame.copy()
            fh, fw = out.shape[:2]
            for (x, y, w, h) in boxes:
                cv2.rectangle(out, (x, y), (x+w, y+h), (0, 200, 100), 2)
            if banner:
                cv2.rectangle(out, (0, 0), (fw, 36), (18, 18, 18), -1)
                cv2.putText(out, banner, (8, 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                            hex2bgr(bcol), 2)
            self._show(out)
        self.root.after(FRAME_MS, self._display_loop)

    # ── Worker ─────────────────────────────────────────────────────────────
    def _worker(self):
        interval = PREVIEW_INTERVAL if self.mode == "register" else DETECT_INTERVAL
        while self._running and self.mode in ("register", "detect"):
            t0    = time.time()
            frame = self._get_frame()
            if frame is not None:
                faces = detect_faces(frame)
                if not faces:
                    with self._lock:
                        self._boxes      = []
                        self._banner     = "No face detected"
                        self._banner_col = MUTED
                    if self.mode == "detect":
                        self.root.after(0, lambda: self._set_status("Scanning…", MUTED))
                else:
                    best = max(faces, key=lambda f: f["box"][2] * f["box"][3])
                    x, y, w, h = [max(0, v) for v in best["box"]]
                    with self._lock:
                        self._boxes = [(x, y, w, h)]

                    if self.mode == "detect":
                        crop = frame[y:y+h, x:x+w]
                        if crop.size > 0:
                            try:
                                emb       = get_embedding(crop)
                                name, sim = match_db(emb)
                                if name:
                                    banner = f"  {name}  ({sim:.2f})"
                                    col    = GREEN
                                    self.root.after(0, lambda n=name, s=sim:
                                        self._set_status(f"Identified: {n}  ({s:.2f})", GREEN))
                                else:
                                    banner = f"  INTRUDER  ({sim:.2f})"
                                    col    = RED
                                    self.root.after(0, lambda:
                                        self._set_status("!! INTRUDER DETECTED !!", RED))
                            except Exception as e:
                                banner, col = f"  Error: {e}", MUTED
                        else:
                            banner, col = "  Face crop failed", MUTED
                        with self._lock:
                            self._banner     = banner
                            self._banner_col = col
                    else:
                        with self._lock:
                            self._banner     = "  Ready — press SPACE to capture"
                            self._banner_col = ORANGE

            time.sleep(max(0.0, interval - (time.time() - t0)))

    # ── Register capture ───────────────────────────────────────────────────
    def _capture_register(self):
        if self.mode != "register" or not self.cap:
            return
        frame = self._get_frame()
        if frame is None:
            return
        faces = detect_faces(frame)
        if not faces:
            messagebox.showwarning("No Face", "No face detected — try again.")
            return
        best = max(faces, key=lambda f: f["box"][2] * f["box"][3])
        x, y, w, h = [max(0, v) for v in best["box"]]
        crop = frame[y:y+h, x:x+w]
        if crop.size == 0:
            messagebox.showwarning("Error", "Face crop too small.")
            return
        try:
            emb = get_embedding(crop)
            save_user(self._reg_name, emb)
            messagebox.showinfo("Registered", f"'{self._reg_name}' saved successfully.")
            self._stop()
        except Exception as e:
            messagebox.showerror("Error", f"Registration failed:\n{e}")

    # ── Helpers ────────────────────────────────────────────────────────────
    def _set_status(self, text, color=TEXT):
        self._status.config(text=text, fg=color)

    def _show(self, bgr):
        cw = max(self.canvas.winfo_width(),  1)
        ch = max(self.canvas.winfo_height(), 1)
        rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil  = Image.fromarray(rgb).resize((cw, ch), Image.LANCZOS)
        imgt = ImageTk.PhotoImage(pil)
        self.canvas.imgtk = imgt
        self.canvas.config(image=imgt)


# ── Entry ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
