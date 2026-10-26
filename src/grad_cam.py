from PIL import Image
from torchvision.models import vgg16

model = vgg16(pretrained=True)
