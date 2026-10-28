import onnxruntime as ort

path = (
    "logs/rsl_rl/lift_kuka_rgb/"
    "2026-10-20_17-12-35/"
    "2026-10-20_17-12-35.onnx"
)

session = ort.InferenceSession(
    path,
    providers=["CPUExecutionProvider"],
)

print("Model loaded successfully!")

print("\nInputs:")
for inp in session.get_inputs():
    print(f"  Name: {inp.name}, Shape: {inp.shape}, Type: {inp.type}")

print("\nOutputs:")
for out in session.get_outputs():
    print(f"  Name: {out.name}, Shape: {out.shape}, Type: {out.type}")

print("\nMetadata:")
print(session.get_modelmeta().custom_metadata_map)

## Inference example
# import numpy as np
# import onnxruntime as ort

# model_path = (
#     "logs/rsl_rl/lift_kuka_rgb/"
#     "2026-10-20_17-12-35/"
#     "2026-10-20_17-12-35.onnx"
# )

# session = ort.InferenceSession(
#     model_path,
#     providers=["CPUExecutionProvider"],
# )

# # Placeholder inputs for testing the model interface only.
# obs = np.zeros((1, 26), dtype=np.float32)
# camera = np.zeros((1, 3, 60, 60), dtype=np.float32)

# actions = session.run(
#     ["actions"],
#     {
#         "obs": obs,
#         "camera": camera,
#     },
# )[0]

# print("Actions:", actions)
# print("Action shape:", actions.shape)
# print("All outputs finite:", np.isfinite(actions).all())
