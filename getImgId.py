path = './metadata/ILSVRC/train/ImageNetV2_boxes.txt'

out = './metadata/ILSVRC/train/image_ids.txt'

# read into pd dataframe
import pandas as pd
df = pd.read_csv(path, sep=',', header=None)
# set first column as index
df.set_index(0, inplace=True)

# get image ids
img_ids = df.index.unique()
# write to a text file separated by new line
with open(out, 'w') as f:
    for img_id in img_ids:
        f.write(img_id + '\n')


