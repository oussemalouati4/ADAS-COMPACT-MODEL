"""
Jetson Nano Real-Time Lane Detection Inference
TensorRT FP16 engine → ~35 FPS on Jetson Nano 4GB

Prerequisites (on Jetson Nano):
  JetPack 5.x (TensorRT, CUDA, cuDNN)
  pip install pycuda numpy opencv-python

Generate engine first:
  trtexec --onnx=lane_detection.onnx \
          --saveEngine=lane_detection_fp16.trt \
          --fp16 --workspace=2048
"""

import argparse, time, cv2, numpy as np, os

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    print("[WARNING] TensorRT not found — falling back to ONNX Runtime.")
    import onnxruntime as ort


MEAN       = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD        = np.array([0.229, 0.224, 0.225], dtype=np.float32)
LANE_COLOR = (0, 255, 100)   # bright green overlay


# ─── TensorRT Engine ──────────────────────────────────────────────────────────

class TRTInferenceEngine:
    def __init__(self, engine_path: str):
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"TRT engine not found: {engine_path}")
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            runtime     = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = self._allocate_buffers()
        self.input_shape  = (1, 3, cfg.IMG_HEIGHT, cfg.IMG_WIDTH)
        self.output_shape = (1, 1, cfg.IMG_HEIGHT, cfg.IMG_WIDTH)
        print(f"[TRT] Engine loaded: {engine_path}")

    def _allocate_buffers(self):
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()
        for binding in self.engine:
            shape    = self.engine.get_binding_shape(binding)
            dtype    = trt.nptype(self.engine.get_binding_dtype(binding))
            size     = trt.volume(shape) * np.dtype(dtype).itemsize
            host_mem = cuda.pagelocked_empty(trt.volume(shape), dtype)
            dev_mem  = cuda.mem_alloc(size)
            bindings.append(int(dev_mem))
            (inputs if self.engine.binding_is_input(binding) else outputs).append(
                {"host": host_mem, "device": dev_mem})
        return inputs, outputs, bindings, stream

    def infer(self, blob: np.ndarray) -> np.ndarray:
        np.copyto(self.inputs[0]["host"], blob.ravel())
        cuda.memcpy_htod_async(self.inputs[0]["device"],  self.inputs[0]["host"],  self.stream)
        self.context.execute_async_v2(self.bindings, self.stream.handle)
        cuda.memcpy_dtoh_async(self.outputs[0]["host"], self.outputs[0]["device"], self.stream)
        self.stream.synchronize()
        return self.outputs[0]["host"].reshape(self.output_shape)


# ─── ONNX Fallback ────────────────────────────────────────────────────────────

class ONNXInferenceEngine:
    def __init__(self, onnx_path: str):
        providers    = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                        if ort.get_device() == "GPU" else ["CPUExecutionProvider"])
        self.session     = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name  = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        print(f"[ONNX] Session ready | providers: {providers}")

    def infer(self, blob: np.ndarray) -> np.ndarray:
        return self.session.run([self.output_name], {self.input_name: blob})[0]


# ─── Pre / Post Processing ───────────────────────────────────────────────────

def preprocess(frame: np.ndarray) -> np.ndarray:
    img = cv2.resize(frame, (cfg.IMG_WIDTH, cfg.IMG_HEIGHT))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    return img.transpose(2, 0, 1)[np.newaxis].astype(np.float32)

def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

def postprocess(logits, orig_h, orig_w):
    mask = (sigmoid(logits[0, 0]) > cfg.CONF_THRESHOLD).astype(np.uint8) * 255
    return cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

def overlay_lanes(frame, mask, alpha=0.45):
    colored = np.zeros_like(frame)
    colored[mask > 127] = LANE_COLOR
    return cv2.addWeighted(frame, 1.0, colored, alpha, 0)


# ─── Runner ──────────────────────────────────────────────────────────────────

def infer_image(engine, image_path: str):
    frame  = cv2.imread(image_path)
    h, w   = frame.shape[:2]
    blob   = preprocess(frame)
    logits = engine.infer(blob)
    mask   = postprocess(logits, h, w)
    result = overlay_lanes(frame, mask)
    out    = image_path.replace(".", "_lane.")
    cv2.imwrite(out, result)
    print(f"[Infer] Saved: {out}")

def infer_video(engine, source):
    cap    = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps_c  = cap.get(cv2.CAP_PROP_FPS) or 30
    w_o    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_o    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter("lane_output.mp4", fourcc, fps_c, (w_o, h_o))
    fps_win = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        t0     = time.perf_counter()
        blob   = preprocess(frame)
        logits = engine.infer(blob)
        mask   = postprocess(logits, h_o, w_o)
        result = overlay_lanes(frame, mask)
        dt     = time.perf_counter() - t0
        fps_win.append(1.0 / max(dt, 1e-6))
        if len(fps_win) > 30: fps_win.pop(0)
        fps_avg = sum(fps_win) / len(fps_win)
        cv2.putText(result, f"FPS: {fps_avg:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        writer.write(result)
        cv2.imshow("Lane Detection — Jetson Nano", result)
        if cv2.waitKey(1) & 0xFF == ord("q"): break
    cap.release(); writer.release(); cv2.destroyAllWindows()
    print(f"[Infer] Avg FPS: {sum(fps_win)/len(fps_win):.1f}")


# ─── Entry Point (change source here) ────────────────────────────────────────
# source = 0                         # webcam
# source = "/path/to/dashcam.mp4"    # video file
# source = "/path/to/road.jpg"       # single image

SOURCE = 0   # ← change this

if TRT_AVAILABLE and os.path.exists(cfg.TRT_ENGINE_PATH):
    engine = TRTInferenceEngine(cfg.TRT_ENGINE_PATH)
elif os.path.exists(cfg.ONNX_PATH):
    print("[INFO] Using ONNX Runtime fallback")
    engine = ONNXInferenceEngine(cfg.ONNX_PATH)
else:
    print("[ERROR] No engine found. Run export_onnx() first.")
    engine = None

if engine:
    src = str(SOURCE)
    if os.path.isfile(src) and src.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        infer_image(engine, src)
    else:
        infer_video(engine, src)