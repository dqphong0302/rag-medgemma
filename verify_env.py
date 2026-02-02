import torch
import torchvision
import openvino as ov
import networkx
import graphrag

print(f"Torch: {torch.__version__}")
print(f"Torchvision: {torchvision.__version__}")
print(f"OpenVINO: {ov.__version__}")
print(f"NetworkX: {networkx.__version__}")

try:
    print(f"GraphRAG version: {graphrag.__version__}") # graphrag might not expose __version__ directly
except:
    print("GraphRAG installed (version check failed)")
