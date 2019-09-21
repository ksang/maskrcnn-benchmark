import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

import requests, os, time
from io import BytesIO
from PIL import Image
import numpy as np
import cv2

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

import argparse

parser = argparse.ArgumentParser(description='MaskRCNN benchmark CPU Utilization test')

parser.add_argument('-c', '--config',
                    default='../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml',
                    help='path to pretrained model config file')
parser.add_argument('-d', '--data', default='witcher3.jpg',
                    help='path to dataset')
parser.add_argument('-i', '--interval', default=0.7, type=float,
                    help='interval in seconds to feed data')
parser.add_argument('-o', '--output', default="",
                    help='output folder to save predictions')
parser.add_argument('-f', '--freq', default=10, type=int,
                    help='output frequency')

def load_from_url(url):
    """
    Given an url of an image, downloads the image and
    returns a PIL image
    """
    response = requests.get(url)
    pil_image = Image.open(BytesIO(response.content)).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image

def load_from_file(path):
    """
    Given path of an image, open it and
    returns a PIL image
    """
    pil_image = Image.open(path).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image

def main():
    args = parser.parse_args()
    cfg.merge_from_file(args.config)
    cfg.merge_from_list(["MODEL.DEVICE", "cuda:0"])

    coco_demo = COCODemo(
        cfg,
        min_image_size=2048,
        confidence_threshold=0.7,
    )

    image = load_from_file(args.data)
    batch_id = 1
    elapsed = 0.
    totaltime = 0.
    while True:
        try:
            start_time = time.time()
            predictions = coco_demo.run_on_opencv_image(image)
            elapsed += time.time()-start_time
            time.sleep(args.interval)
            totaltime += time.time()-start_time
            if batch_id % args.freq == 0:
                print("Batch #{}: inference time: {:>3.3f}s, per image: {:>3.4f}s, total time: {:>3.3f}s inference percent: {:>3.0f}%".format(
                    batch_id, elapsed, elapsed/args.freq, totaltime,
                    100.*elapsed/totaltime))
                if args.output:
                    if not os.path.exists(args.output):
                        os.makedirs(args.output)
                    filename = os.path.join(args.output, 'prediction_{}.jpg'.format(batch_id))
                    print("=> Saving prediction to {}".format(filename))
                    cv2.imwrite(filename, predictions)
                elapsed = 0.
                totaltime = 0.
            batch_id+=1
        except KeyboardInterrupt:
            print("Interrupted.")
            break

if __name__ == '__main__':
    main()
