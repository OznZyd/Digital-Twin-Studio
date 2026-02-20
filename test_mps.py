import torch

if torch.backends.mps.is_available():
    print("M3 Prp MPS Active!")
else:
    print("ERROR: CPU Using!")