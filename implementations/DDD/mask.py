import cv2
import torch
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101


def make_deeplab(device):
  deeplab = deeplabv3_resnet101(pretrained=True).to(device)
  deeplab.eval()
  return deeplab


def apply_deeplab(deeplab, img, device):
  input_tensor = deeplab_preprocess(img)
  input_batch = input_tensor.unsqueeze(0)
  with torch.no_grad():
    output = deeplab(input_batch.to(device))['out'][0]
  output_predictions = output.argmax(0).cpu().numpy()
  return (output_predictions == 15)


device = torch.device("cuda")
deeplab = make_deeplab(device)
print("here")
for i in range(10, 559):
  print("Current Image: ", i)
  path = "./images/"
  filename = f'{i}.png'
  img_orig = cv2.imread(path + filename, 1)
  k = min(1.0, 1024 / max(img_orig.shape[0], img_orig.shape[1]))
  img = cv2.resize(img_orig, None, fx=k, fy=k, interpolation=cv2.INTER_LANCZOS4)

  deeplab_preprocess = transforms.Compose(
      [
          transforms.ToTensor(),
          transforms.Normalize(
              mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
          ),
      ]
  )

  mask = apply_deeplab(deeplab, img, device)
  mask.dtype = 'uint8'
  mask *= 255
  cv2.imwrite(f'./images/{filename[:-4]}_masked.png', mask)
