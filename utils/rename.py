import os
class BatchRename():
  def __init__(self):
    self.path = '/home/boryant/experiment_2/input/'
  def rename(self):
    filelist = os.listdir(self.path)
    total_num = len(filelist)
    i = 1
    for item in filelist:
      if item.endswith('.png'):
        src = os.path.join(os.path.abspath(self.path), item)
        item_part = item.split(".", 1)[0]
        # dst = os.path.join(os.path.abspath(self.path), item.replace(item_part,  str(i)))
        dst = os.path.join(os.path.abspath(self.path), item.replace(item_part, str(i)))
        i += 1
        try:
          os.rename(src, dst)
          print('converting %s to %s ...' % (src, dst))
        except:
          continue
    print('total %d to rename & converted %d jpgs' % (total_num, i))
if __name__ == '__main__':
  demo = BatchRename()
  demo.rename()
