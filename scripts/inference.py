from pathlib import Path
from time import perf_counter

import cv2
import pkg_resources
import torch

from scaledyolov4.scaled_yolov4 import ScaledYOLOV4


imgpath = Path('test.jpg')
if not imgpath.is_file():
    raise AssertionError(f'{str(imgpath)} not found')

yolov4 = ScaledYOLOV4(
    weights=pkg_resources.resource_filename('scaledyolov4', 'weights/yolov4-p6_-state.pt'),
    cfg=pkg_resources.resource_filename('scaledyolov4', 'configs/yolov4-p6.yaml'),
    # classes_path=pkg_resources.resource_filename('scaledyolov4', 'data/coco.yaml'),
    bgr=True,
    gpu_device=0,
    model_image_size=608,
    max_batch_size=1,
    half=True,
    same_size=True
)

img = cv2.imread(str(imgpath))
bs = 5
imgs = [img for _ in range(bs)]

n = 30
dur = 0
for i in range(n):
    torch.cuda.synchronize()
    tic = perf_counter()
    dets = yolov4.detect_get_box_in(imgs, box_format='ltrb', classes=None, buffer_ratio=0.0)[0]
    # print('detections: {}'.format(dets))
    torch.cuda.synchronize()
    toc = perf_counter()
    if i > 5:
        dur += toc - tic
print(f'Average time taken: {(dur/n):0.3f}s')

cv2.namedWindow('output', cv2.WINDOW_NORMAL)
draw_frame = img.copy()
for det in dets:
    # print(det)
    bb, score, class_ = det 
    l, t, r, b = bb
    cv2.rectangle(draw_frame, (l, t), (r, b), (255, 255, 0), 1)
    cv2.putText(draw_frame, class_, (l, t-8), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0))

cv2.imwrite('test_out.jpg', draw_frame)
cv2.imshow('output', draw_frame)
cv2.waitKey(0)
