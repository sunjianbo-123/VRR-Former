import random
from PIL import Image
from PIL import ImageFilter



# Open the reflection image and convert it to RGBA
reflection_image = Image.open('./RealData_1488.jpg').convert("RGBA")




# 将reflection_image像素值为255的区域变为透明  其余区域透明度设置为50%
datas = reflection_image.getdata()
newData = []
for item in datas:
    # If all RGB values are 255 (white), then set alpha to 0 (100% transparent)
    if item[0] > 250 and item[1] > 250 and item[2] > 250:
        newData.append((255, 255, 255, 0))
    else:
        # Else set alpha to 128 (50% transparent)
        newData.append((item[0], item[1], item[2], 128))

# Update the image data
reflection_image.putdata(newData)
reflection_image.show()





