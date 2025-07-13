import torch
model = torch.load('./weights/best.pt', map_location='cuda', weights_only=False)['model'].float()
