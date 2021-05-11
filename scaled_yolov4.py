from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch
import yaml

from models.experimental import attempt_load
from utils.torch_utils import intersect_dicts
from utils.general import scale_coords, non_max_suppression, check_img_size


class Scaled_YOLOV4(object):
    THIS_DIR = Path(__file__).resolve().parent
    _defaults = {
        "weights": f"{THIS_DIR}/weights/yolov4-p5_.pt",
        "classes_path": f"{THIS_DIR}/data/coco.yaml",
        "thresh": 0.4,
        "nms_thresh": 0.5,
        "model_image_size": 608,
        "max_batch_size": 4,
        "half": True,
        "same_size": False
    }

    def __init__(self, bgr=True, gpu_device=0, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # update with user overrides

        self.bgr = bgr
        self.device = self._select_device(str(gpu_device))
        self.class_names = self._get_class(self.classes_path)

        self.model = attempt_load(self.weights, map_location=self.device)
        self.model = self.model.to(self.device)
        if self.device == torch.device('cpu'):
            self.half = False
        if self.half:
            self.model.half()

        self.model_image_size = check_img_size(self.model_image_size, s=self.model.stride.max())

        # warm up
        self._detect([np.zeros((10,10,3), dtype=np.uint8)])
        # self._detect([np.zeros((10,10,3), dtype=np.uint8), np.zeros((10,10,3), dtype=np.uint8), np.zeros((10,10,3), dtype=np.uint8), np.zeros((10,10,3), dtype=np.uint8)])
        print('Warmed up!')

    @staticmethod
    def _select_device(device):
        cpu_request = device.lower() == 'cpu'
        if not cpu_request and not device.isnumeric():  # if device requested other than 'cpu'
            device = device.split(':')[-1]
        cuda = False if cpu_request else torch.cuda.is_available()
        print(f'Using CUDA device {device}')
        return torch.device(f'cuda:{device}' if cuda else 'cpu')

    @staticmethod
    def _get_class(classes_path):
        with open(classes_path) as f:
            data_dict = yaml.load(f, Loader=yaml.FullLoader)
        num_classes = int(data_dict['nc'])
        class_names = data_dict['names']
        assert len(class_names) == num_classes, f'{len(class_names)} names found for nc={num_classes} dataset in {classes_path}'
        return class_names

    def _classname_to_idx(self, classname):
        return self.class_names.index(classname)

    def _detect(self, list_of_imgs):
        if self.bgr:
            list_of_imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in list_of_imgs]

        resized = [self._letterbox(img, new_shape=self.model_image_size, auto=self.same_size)[0] for img in list_of_imgs]
        images = np.stack(resized, axis=0)
        images = np.divide(images, 255, dtype=np.float32)
        images = np.ascontiguousarray(images.transpose(0, 3, 1, 2))
        input_shapes = [img.shape for img in images]
        images = torch.from_numpy(images)
        images = images.to(self.device)

        if self.half:
            images = images.half()

        batches = []
        for i in range(0, len(images), self.max_batch_size):
            these_imgs = images[i:i+self.max_batch_size]
            batches.append(these_imgs)

        preds = []
        with torch.no_grad():
            for batch in batches:
                features = self.model(batch)[0]
                preds.append(features)

        predictions = torch.cat(preds, dim=0)

        return predictions, input_shapes

    def detect_get_box_in(self, images, box_format='ltrb', classes=None, buffer_ratio=0.0):
        '''
        Params
        ------
        - images : ndarray-like or list of ndarray-like
        - box_format : string of characters representing format order, where l = left, t = top, r = right, b = bottom, w = width and h = height
        - classes : list of string, classes to focus on
        - buffer : float, proportion of buffer around the width and height of the bounding box

        Returns
        -------
        if one ndarray given, this returns a list (boxes in one image) of tuple (box_infos, score, predicted_class),
        
        else if a list of ndarray given, this return a list (batch) containing the former as the elements,

        where,
            - box_infos : list of floats in the given box format
            - score : float, confidence level of prediction
            - predicted_class : string

        '''
        single = False
        if isinstance(images, list):
            if len(images) <= 0 : 
                return None
            else:
                assert all(isinstance(im, np.ndarray) for im in images)
        elif isinstance(images, np.ndarray):
            images = [ images ]
            single = True

        res, input_shapes = self._detect(images)
        frame_shapes = [image.shape for image in images]
        all_dets = self._postprocess(res, input_shapes=input_shapes, frame_shapes=frame_shapes, box_format=box_format, classes=classes, buffer_ratio=buffer_ratio)

        if single:
            return all_dets[0]
        else:
            return all_dets

    def get_detections_dict(self, frames, classes=None, buffer_ratio=0.0):
        '''
        Params: frames, list of ndarray-like
        Returns: detections, list of dict, whose key: label, confidence, t, l, w, h
        '''
        if frames is None or len(frames) == 0:
            return None
        all_dets = self.detect_get_box_in( frames, box_format='tlbrwh', classes=classes, buffer_ratio=buffer_ratio )
        
        all_detections = []
        for dets in all_dets:
            detections = []
            for tlbrwh,confidence,label in dets:
                top, left, bot, right, width, height = tlbrwh
                detections.append( {'label':label,'confidence':confidence,'t':top,'l':left,'b':bot,'r':right,'w':width,'h':height} ) 
            all_detections.append(detections)
        return all_detections

    @staticmethod
    def _letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
        # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

        return img, ratio, (dw, dh)

    def _postprocess(self, boxes, input_shapes, frame_shapes, box_format='ltrb', classes=None, buffer_ratio=0.0):
        class_idxs = [self._classname_to_idx(name) for name in classes] if classes is not None else None
        preds = non_max_suppression(boxes, self.thresh, self.nms_thresh, classes=class_idxs)

        detections = []
        for i, frame_bbs in enumerate(preds):
            if frame_bbs is None:
                detections.append([])
                continue

            im_height, im_width, _ = frame_shapes[i]
            
            # Rescale preds from input size to frame size
            frame_bbs[:, :4] = scale_coords(input_shapes[i][1:], frame_bbs[:, :4], frame_shapes[i]).round()

            frame_dets = []
            for *xyxy, cls_conf, cls_id in frame_bbs:
                cls_conf = float(cls_conf)
                cls_name = self.class_names[int(cls_id)]

                left = int(xyxy[0])
                top = int(xyxy[1])
                right = int(xyxy[2])
                bottom = int(xyxy[3])
                
                width = right - left + 1
                height = bottom - top + 1
                width_buffer = width * buffer_ratio
                height_buffer = height * buffer_ratio

                top = max(0.0, top - 0.5*height_buffer)
                left = max(0.0, left - 0.5*width_buffer)
                bottom = min(im_height - 1.0, bottom + 0.5*height_buffer)
                right = min(im_width - 1.0, right + 0.5*width_buffer)

                box_infos = []
                for c in box_format:
                    if c == 't':
                        box_infos.append( int(round(top)) ) 
                    elif c == 'l':
                        box_infos.append( int(round(left)) )
                    elif c == 'b':
                        box_infos.append( int(round(bottom)) )
                    elif c == 'r':
                        box_infos.append( int(round(right)) )
                    elif c == 'w':
                        box_infos.append( int(round(width+width_buffer)) )
                    elif c == 'h':
                        box_infos.append( int(round(height+height_buffer)) )
                    else:
                        assert False,'box_format given in detect unrecognised!'
                assert len(box_infos) > 0 ,'box infos is blank'

                detection = (box_infos, cls_conf, cls_name)
                frame_dets.append(detection)
            detections.append(frame_dets)

        return detections


if __name__ == '__main__':
    import cv2
    from pathlib import Path

    imgpath = 'test.jpg'
    assert Path(imgpath).is_file() , 'image not found'

    yolov4 = Scaled_YOLOV4( 
        bgr=True,
        gpu_device=0,
        model_image_size=608,
        max_batch_size=1,
        half=True,
        same_size=True
    )

    img = cv2.imread(imgpath)
    bs = 5
    imgs = [ img for _ in range(bs) ]

    n = 30
    dur = 0
    for i in range(n):
        torch.cuda.synchronize()
        tic = perf_counter()
        dets = yolov4.detect_get_box_in(imgs, box_format='ltrb', classes=None, buffer_ratio=0.0)[0]
        # print('detections: {}'.format(dets))
        torch.cuda.synchronize()
        toc = perf_counter()
        if i>5:
            dur += toc - tic
    print('Average time taken: {:0.3f}s'.format(dur/n))

    cv2.namedWindow('output', cv2.WINDOW_NORMAL)
    draw_frame = img.copy()
    for det in dets:
        # print(det)
        bb, score, class_ = det 
        l,t,r,b = bb
        cv2.rectangle(draw_frame, (l,t), (r,b), (255,255,0), 1)
        cv2.putText(draw_frame, class_, (l, t-8), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0))
    
    cv2.imwrite('test_out.jpg', draw_frame)
    cv2.imshow('output', draw_frame)
    cv2.waitKey(0)
