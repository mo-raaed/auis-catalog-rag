import torch, sys

print("python executable:", sys.executable)
print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
else:
    print("device: no gpu visible to torch")
