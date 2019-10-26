import sidekick
from PIL import Image
import sys


url = sys.argv[1]
token = sys.argv[2]
filename = sys.argv[3]
client = sidekick.Deployment(url=url , token=token)
image = Image.open(filename)
banana = client.predict(image=image)
print(banana)