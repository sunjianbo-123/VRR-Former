# -*- coding: UTF-8 -*-
import os
from PIL import Image
import sys




"""

txtfilepath = '/home/boryant/PycharmProjects/MyProjects/ReflectionRemovalCodes/SRRNs/flash-reflection-removal_tfversion' \
			  '/data/real_world/val/others/'
saveBasePath = "/home/boryant/PycharmProjects/MyProjects/ReflectionRemovalCodes/perceptual-reflection-removal/tools/"

temp_xml = os.listdir(txtfilepath)
total_xml = []
for xml in temp_xml:
	if xml.endswith("_R.jpg"):
		total_xml.append(xml)


ftrainval = open(os.path.join(saveBasePath, 'jpg.txt'), 'w')

i = 0
print(len(total_xml))
for i in range(len(total_xml)):
	name = total_xml[i][:-4] + '.jpg' + "\n"
	ftrainval.write(name)

"""





































data = []
for line in open("/home/boryant/PycharmProjects/MyProjects/ReflectionRemovalCodes/perceptual-reflection-removal/tools/jpg.txt", "r"):                     # 设置文件对象并读取每一行文件
    data.append(line)

num = 1
for a in data:
  im = Image.open('/home/boryant/PycharmProjects/MyProjects/ReflectionRemovalCodes/SRRNs/flash-reflection-removal_tfversion' \
			  '/data/real_world/val/others/{}'.format(a[:-1]))
  im.save('/home/boryant/Pictures/flash_reflection_removal/real/val/R/{}'.format(a[:-1]))
  im.close()
  print("copyed {} pictures".format(num))
  num += 1




















































