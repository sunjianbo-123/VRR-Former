import os
class BatchRename():
  def __init__(self):
    self.path = '/home/boryant/download/ReflectionRemovalDatas10672/SyntheticData10693/ReflectionLayer10693/2_flash-reflection-removal7797/synthetic7326' \
                '/with_syn_reflection7326/train_r5853/'
  def rename(self):
    filelist = os.listdir(self.path)
    total_num = len(filelist)
    i = 4840
    for item in filelist:
      if item.endswith('.jpg'):
        src = os.path.join(os.path.abspath(self.path), item)
        dst = os.path.join(os.path.abspath(self.path), "SynData_" + str(i) + '.jpg')
        try:
          os.rename(src, dst)
          print ('converting %s to %s ...' % (src, dst))
          i = i + 1
        except:
          continue
    print ('total %d to rename & converted %d jpgs' % (total_num, i))
if __name__ == '__main__':
  demo = BatchRename()
  demo.rename()

