import torch
import onnx
import onnxruntime as ort
import numpy as np


def export_onnx():
    print("\n[Export] Loading best model checkpoint...")
    model = build_model(pretrained=False)
    ck    = torch.load(cfg.BEST_MODEL_PATH, map_location="cpu")
    model.load_state_dict(ck["model"])
    model.eval()

    dummy = torch.randn(1, 3, cfg.IMG_HEIGHT, cfg.IMG_WIDTH)

    print(f"[Export] Exporting to ONNX: {cfg.ONNX_PATH}")
    torch.onnx.export(
        model, dummy, cfg.ONNX_PATH,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["lane_mask"],
        dynamic_axes={"image": {0: "batch"}, "lane_mask": {0: "batch"}},
        verbose=False,
    )

    # Validate ONNX graph
    onnx_model = onnx.load(cfg.ONNX_PATH)
    onnx.checker.check_model(onnx_model)
    print("  ✓ ONNX graph valid")

    # Compare PyTorch vs ONNX outputs
    sess = ort.InferenceSession(cfg.ONNX_PATH, providers=["CPUExecutionProvider"])
    with torch.no_grad():
        pt_out = torch.sigmoid(model(dummy)).numpy()
    ort_out = sess.run(["lane_mask"], {"image": dummy.numpy()})[0]
    ort_out = 1 / (1 + np.exp(-ort_out))   # sigmoid

    max_diff = np.abs(pt_out - ort_out).max()
    print(f"  ✓ Max output diff PyTorch vs ONNX: {max_diff:.6f}")
    assert max_diff < 1e-4, "ONNX outputs differ too much!"

    print(f"\n[Export] Done! → {cfg.ONNX_PATH}")
    print("\nNext: copy lane_detection.onnx to your Jetson Nano and run:")
    print(f"  trtexec --onnx={cfg.ONNX_PATH} --saveEngine={cfg.TRT_ENGINE_PATH} --fp16 --workspace=2048")


export_onnx()