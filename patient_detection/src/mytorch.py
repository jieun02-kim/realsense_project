import torch
print(torch.version.cuda)       # PyTorch가 빌드된 CUDA 버전
print(torch.cuda.is_available()) # CUDA 사용 가능 여부 (True/False)
