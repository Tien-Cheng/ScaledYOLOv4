import cv2
import numpy as np
import pkg_resources
import torch
import yaml

from scaledyolov4.models.experimental import attempt_load_state_dict
from scaledyolov4.models.yolo import Model
from scaledyolov4.utils.general import scale_coords, non_max_suppression, check_img_size, process_raw_data


class ScaledYOLOV4:
    _defaults = {
        'thresh': 0.4,
        'nms_thresh': 0.5,
        'model_image_size': 608,
        'max_batch_size': 4,
        'half': True,
        'same_size': False,
        'weights': pkg_resources.resource_filename('scaledyolov4', 'weights/yolov4-p5_-state.pt'),
        'cfg': pkg_resources.resource_filename('scaledyolov4', 'configs/yolov4-p5.yaml'),
        'classes_path': pkg_resources.resource_filename('scaledyolov4', 'data/coco.yaml'),
    }

    def __init__(self, bgr=True, gpu_device=0, **kwargs):
        self.__dict__.update(self._defaults)  # set up default values
        self.__dict__.update(kwargs)  # update with user overrides

        self.bgr = bgr
        self.device, self.device_num = self._select_device(str(gpu_device))
        self.class_names = self._get_class(self.classes_path)

        model = Model(self.cfg)
        self.model = attempt_load_state_dict(model, self.weights, map_location=self.device)
        self.model.to(self.device)
        self.model.eval()
        if self.device == torch.device('cpu'):
            self.half = False
        if self.half:
            self.model.half()

        gs = max(self.model.stride.max(), 32)
        self.model_image_size = check_img_size(self.model_image_size, s=gs)

        # warm up
        self._detect([np.zeros((10, 10, 3), dtype=np.uint8)])
        # self._detect([np.zeros((10,10,3), dtype=np.uint8), np.zeros((10,10,3), dtype=np.uint8), np.zeros((10,10,3), dtype=np.uint8), np.zeros((10,10,3), dtype=np.uint8)])
        print('Warmed up!')

    @staticmethod
    def _select_device(device):
        cpu_request = device.lower() == 'cpu'
        if cpu_request:
            print('Using CPU')
            return torch.device('cpu')
        if not device.isnumeric():
            device = device.split(':')[-1]
        print(f'Using CUDA device {device}')
        return torch.device(f'cuda:{device}'), int(device)

    @staticmethod
    def _get_class(classes_path):
        with open(classes_path) as f:
            data_dict = yaml.safe_load(f)
        num_classes = int(data_dict['nc'])
        class_names = data_dict['names']
        if len(class_names) != num_classes:
            raise AssertionError(f'{len(class_names)} names found for nc={num_classes} dataset in {classes_path}')
        return class_names

    def classname_to_idx(self, classname):
        return self.class_names.index(classname)

    @torch.no_grad()
    def _detect(self, list_of_imgs):
        if self.bgr:
            list_of_imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in list_of_imgs]

        resized = [self._letterbox(img, new_shape=self.model_image_size, auto=self.same_size)[0] for img in list_of_imgs]
        images = np.stack(resized, axis=0)
        images = np.divide(images, 255, dtype=np.float32)
        images = np.ascontiguousarray(images.transpose(0, 3, 1, 2))
        input_shapes = [img.shape for img in images]
        images = torch.from_numpy(images)

        if self.half:
            images = images.half()

        batches = []
        for i in range(0, len(images), self.max_batch_size):
            these_imgs = images[i:i+self.max_batch_size]
            batches.append(these_imgs)

        preds = []
        with torch.cuda.device(self.device_num):
            for batch in batches:
                batch = batch.to(self.device)
                features = self.model(batch)[0]
                preds.append(features.cpu())

        predictions = torch.cat(preds, dim=0)

        return predictions, input_shapes

    def detect_get_box_in(self, images, box_format='ltrb', classes=None, buffer_ratio=0.0, raw=False):
        '''
        Parameters
        ----------
        images : ndarray or List[ndarray]
            ndarray-like for single image or list of ndarray-like
        box_format : str, optional
            string of characters representing format order, where l = left, t = top, r = right, b = bottom, w = width and h = height
        classes : List[str], optional
            classes to focus on
        buffer_ratio : float, optional
            proportion of buffer around the width and height of the bounding box
        raw : bool, optional
            return raw inferences instead of detections after postprocessing
        
        Returns
        ------
        If raw is False:
            If one ndarray given, this returns a list (boxes in one image) of tuple (box_infos, score, predicted_class),
            else if a list of ndarray given, this return a list (batch) containing the former as the elements.
            box_infos : List[float]
                according to the given box format
            score : float
                confidence level of prediction
            predicted_class : string

        If raw is True:
            If one ndarray given, this returns an ndarray of size (max_det, num_classes+5)
            [x1, y1, x2, y2, objectness, *cls_probabilities], else if a list of ndarray given, this return a list (batch)
            containing the former as the elements. max_det is 300 by default.
        '''
        single = False
        if isinstance(images, list):
            if len(images) <= 0:
                return None
            else:
                if not all(isinstance(im, np.ndarray) for im in images):
                    raise AssertionError('all images must be np arrays')
        elif isinstance(images, np.ndarray):
            images = [images]
            single = True

        res, input_shapes = self._detect(images)
        frame_shapes = [image.shape for image in images]
        if raw:
            all_dets = self._postprocess_raw(res, input_shapes=input_shapes, frame_shapes=frame_shapes, box_format=box_format)
        else:
            all_dets = self._postprocess(res, input_shapes=input_shapes, frame_shapes=frame_shapes, box_format=box_format, classes=classes, buffer_ratio=buffer_ratio)

        if single:
            return all_dets[0]
        else:
            return all_dets

    def get_detections_dict(self, frames, classes=None, buffer_ratio=0.0):
        '''
        Parameters
        ----------
        frames : List[ndarray]
            list of input images
        classes : List[str], optional
            classes to focus on
        buffer_ratio : float, optional
            proportion of buffer around the width and height of the bounding box

        Returns
        -------
        List[dict]
            list of detections for each frame with keys: label, confidence, t, l, w, h
        '''

        if frames is None or len(frames) == 0:
            return None
        all_dets = self.detect_get_box_in(frames, box_format='tlbrwh', classes=classes, buffer_ratio=buffer_ratio)
        
        all_detections = []
        for dets in all_dets:
            detections = []
            for tlbrwh, confidence, label in dets:
                top, left, bot, right, width, height = tlbrwh
                detections.append({'label': label, 'confidence': confidence, 't': top, 'l': left, 'b': bot, 'r': right, 'w': width, 'h': height})
            all_detections.append(detections)
        return all_detections

    @staticmethod
    def _letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scalefill=False, scaleup=True):
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
            dw, dh = np.mod(dw, 128), np.mod(dh, 128)  # wh padding
        elif scalefill:  # stretch
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
        class_idxs = [self.classname_to_idx(name) for name in classes] if classes is not None else None
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
                    if c not in [*'tlbrwh']:
                        raise AssertionError('box_format given in detect unrecognised!')
                    elif c == 't':
                        box_infos.append(int(round(top)))
                    elif c == 'l':
                        box_infos.append(int(round(left)))
                    elif c == 'b':
                        box_infos.append(int(round(bottom)))
                    elif c == 'r':
                        box_infos.append(int(round(right)))
                    elif c == 'w':
                        box_infos.append(int(round(width+width_buffer)))
                    elif c == 'h':
                        box_infos.append(int(round(height+height_buffer)))
                if not len(box_infos) > 0:
                    raise AssertionError('box infos is blank')

                detection = (box_infos, cls_conf, cls_name)
                frame_dets.append(detection)
            detections.append(frame_dets)

        return detections

    def _postprocess_raw(self, boxes, input_shapes, frame_shapes, box_format='ltrb'):
        if box_format != 'ltrb':
            raise AssertionError('box_format can only be ltrb for raw outputs!')

        preds = process_raw_data(boxes)

        detections = []
        for i, frame_bbs in enumerate(preds):
            if frame_bbs is None:
                detections.append([])
                continue

            im_height, im_width, _ = frame_shapes[i]

            # Rescale preds from input size to frame size
            frame_bbs[:, :4] = scale_coords(input_shapes[i][1:], frame_bbs[:, :4], frame_shapes[i]).round()
            detections.append(frame_bbs.numpy())

        return detections
