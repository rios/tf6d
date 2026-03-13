import pandas as pd
import pose_utils.vis_utils as vis_utils
import cv2
import json
from PIL import Image
import numpy as np
import os


def load_json_file(file_path):
    """Load a JSON file and return its content."""
    with open(file_path, 'r') as f:
        return json.load(f)


# Example usage:
dataset_name = 'lmo'
# MASt3R401-tless-test_bop.csv
file_name = 'MASt3R40Filter_%s-test_bop' % (dataset_name)
input_file_path = './logs/results_fastsam_final/%s.csv' % (file_name)
gt_path = "./database/gts/test_gts/%s_bop_test_gt_fastsam_filter.json" % (dataset_name)
data_gt = load_json_file(gt_path)  # load fastSAM
dataset_path = "./data/bop/%s/" % (dataset_name)

save_path = 'logs/vis/'

# Load the input CSV file
data = pd.read_csv(input_file_path)
data['im_id'] = data['im_id'].astype(str).apply(lambda x: x.replace('.tif', '') if '.tif' in x else x)

# (2,803) (2, 3), (2, 1212), (2. 991)
scene_id = 2
image_id = 803 # 112
filtered_data = data[(data['scene_id'] == scene_id) & (data['im_id'] == f'{int(image_id):01d}')]
gt_data = data_gt[f'{int(scene_id):06d}_{image_id}']

image_name = gt_data[0]['img_name']
image_path = dataset_path + image_name
img_data = Image.open(image_path)
h, w = img_data.size

for index, row in enumerate(filtered_data.iterrows()):
    # if index > 3:
    #     break

    R = row[1]['R']
    t = row[1]['t']

    data_list = list(map(float, R.split()))
    R = np.array(data_list).reshape(3, 3)

    data_list = list(map(float, t.split()))
    t = np.array(data_list).reshape(3, ) / 1000

    t_gt = np.array(gt_data[index]['cam_t_m2c']) / 1000
    R_gt = np.array(gt_data[index]['cam_R_m2c']).reshape((3, 3))


    cam_K = np.array(gt_data[index]['cam_K']).reshape((3, 3))

    img_data = vis_utils.create_debug_image(R, t, R_gt, t_gt,  np.asarray(img_data), cam_K,
                                                 gt_data[index]['model_info'], 0.001,
                                                 image_shape=(w, h),
                                                 colEst=(0, 255, 0))

img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
cv2.imwrite(os.path.join(save_path, 'demo.png'), img_data)

print()
