import torch
from rtx import RT2X


# usage
img = torch.randn(1, 3, 256, 256)
text = torch.randint(0, 20000, (1, 1024))

model = RT2X()
output = model(img, text)
print(output)
