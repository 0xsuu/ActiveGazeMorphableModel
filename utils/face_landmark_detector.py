
# Code modified from pytorch_face_landmark repository on GitHub, original author is:
# Cunjian Chen (ccunjian@gmail.com), Feb. 2021

import cv2
import numpy as np
import torch

from third_party.pytorch_face_landmark.utils.align_trans import get_reference_facial_points
from third_party.pytorch_face_landmark.models.basenet import MobileNet_GDConv
from third_party.pytorch_face_landmark.Retinaface import Retinaface
from third_party.pytorch_face_landmark.common.utils import BBox, drawLandmark_multiple


class FaceLandmarkDetector(object):
    def __init__(self):
        self.mean = np.asarray([0.485, 0.456, 0.406])
        self.std = np.asarray([0.229, 0.224, 0.225])

        # crop_size = 112
        # scale = crop_size / 112.
        # reference = get_reference_facial_points(default_square=True) * scale

        self.out_size = 224
        model = MobileNet_GDConv(136).cuda()
        model = torch.nn.DataParallel(model)
        # Download model from https://drive.google.com/file/d/1Le5UdpMkKOTRr1sTp4lwkw8263sbgdSe/view?usp=sharing.
        checkpoint = torch.load(
            "../third_party/pytorch_face_landmark/checkpoint/mobilenet_224_model_best_gdconv_external.pth.tar",
            map_location="cuda")

        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        self.model = model

        self.retina_face = Retinaface.Retinaface()

    def process_face(self, image):
        faces = self.retina_face(image)
        if len(faces) == 0:
            return None, None
        
        if len(faces) == 1:
            face = faces[0]
        else:
            face = max(faces, key=lambda x: x[4])

        if face[4] < 0.9:
            return None, None
        x1, y1, x2, y2 = face[0], face[1], face[2], face[3]
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        size = int(min([w, h])*1.2)
        cx = x1 + w//2
        cy = y1 + h//2
        x1 = cx - size//2
        x2 = x1 + size
        y1 = cy - size//2
        y2 = y1 + size

        dx = max(0, -x1)
        dy = max(0, -y1)
        x1 = max(0, x1)
        y1 = max(0, y1)

        height, width, _ = image.shape
        edx = max(0, x2 - width)
        edy = max(0, y2 - height)
        x2 = min(width, x2)
        y2 = min(height, y2)
        new_bbox = list(map(int, [x1, x2, y1, y2]))
        new_bbox = BBox(new_bbox)
        cropped = image[new_bbox.top:new_bbox.bottom, new_bbox.left:new_bbox.right]
        if dx > 0 or dy > 0 or edx > 0 or edy > 0:
            cropped = cv2.copyMakeBorder(cropped, int(dy), int(edy), int(dx), int(edx), cv2.BORDER_CONSTANT, 0)
        cropped_face = cv2.resize(cropped, (self.out_size, self.out_size))

        if cropped_face.shape[0] <= 0 or cropped_face.shape[1] <= 0:
            raise ValueError("Cropping failed.")

        test_face = cropped_face.copy()
        test_face = test_face / 255.0
        test_face = (test_face - self.mean) / self.std
        test_face = test_face.transpose((2, 0, 1))
        test_face = test_face.reshape((1,) + test_face.shape)
        input_tensor = torch.from_numpy(test_face).float()
        input_tensor = torch.autograd.Variable(input_tensor)

        landmark = self.model(input_tensor).cpu().numpy()
        landmark = landmark.reshape(-1, 2)
        landmark = new_bbox.reprojectLandmark(landmark)

        left_eyebrow_lm = landmark[17:22]
        right_eyebrow_lm = landmark[22:27]
        nose_ridge_lm = landmark[27:31]
        nose_bottom_lm = landmark[31:36]
        left_eye_lm = landmark[36:42]
        right_eye_lm = landmark[42:48]

        return list(map(int, [x1, y1, x2, y2])), \
            [left_eyebrow_lm, right_eyebrow_lm, nose_ridge_lm, nose_bottom_lm, left_eye_lm, right_eye_lm]
