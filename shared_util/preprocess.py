import torch
import face_alignment as FAN
import numpy as np
import cv2
from skimage.transform import SimilarityTransform, PiecewiseAffineTransform, warp
import os
import matplotlib.pyplot as plt

FAN_CHECKPOINT = ""
STANDARD_FACE_PATH = "standard_face_68.npy"



class PreProcess:
    def __init__(self, ops):
        """
        :param ops: Options
        """
        # Change the current working directory to the current_file_directory/face_alignment
        current_working_directory = os.getcwd()
        current_file_directory = os.path.dirname(__file__)
        os.chdir(os.path.join(current_file_directory, "face_alignment"))

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.image_size = ops.image_size
        self.image_scale_to_before_crop = ops.image_scale_to_before_crop
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # Histogram normalizer

        # load FAN landmark detector including SFD face detector
        self.FAN = FAN.FaceAlignment(FAN.LandmarksType._2D, flip_input=True, device=self.device, check_point_path=FAN_CHECKPOINT)
        self.face_detector = self.FAN.get_landmarks_from_image

        self.mean_lmks = np.load(STANDARD_FACE_PATH)
        self.mean_lmks = self.mean_lmks * 155 / self.mean_lmks.max()
        self.mean_lmks[:, 1] += 15

        # Restore the current working directory
        os.chdir(current_working_directory)
       


    @staticmethod
    def crop_image(frame, bbox):
        fh, fw = frame.shape[:2]
        bl, bt, br, bb = bbox
        fh, fw, bl, bt, br, bb = int(fh), int(fw), int(bl), int(bt), int(br), int(bb)

        a_slice = frame[max(0, min(bt, fh)):min(fh, max(bb, 0)), max(0, min(bl, fw)):min(fw, max(br, 0)), :]
        new_image = np.zeros((bb - bt, br - bl, 3), dtype=np.float32)
        new_image[max(0, min(bt, fh)) - bt:min(fh, max(bb, 0)) - bt, max(0, min(bl, fw)) - bl:min(fw, max(br, 0)) - bl,
                  :] = a_slice

        h, w = new_image.shape[:2]
        m = max(h, w)
        square_image = np.zeros((m, m, 3), dtype=np.float32)
        square_image[(m - h) // 2:h + (m - h) // 2, (m - w) // 2:w + (m - w) // 2, :] = new_image
        return square_image



    @staticmethod
    def similarity_transform(image, landmarks):
        # anchor coordinate are based on the 240x320 resolution and need to be scaled accordingly for different size images.
        anchor_scale = 320 / image.shape[1]
        anchor = np.array([[110, 71], [210, 71], [160, 170]], np.float32) / anchor_scale
        idx = [36, 45, 57]
        tform = SimilarityTransform()
        tform.estimate(landmarks[idx, :], anchor)
        sim_mat = tform.params[:2, :]
        dst = cv2.warpAffine(image, sim_mat, (image.shape[1], image.shape[0]))
        dst_lmks = np.matmul(np.concatenate((landmarks, np.ones((landmarks.shape[0], 1))), 1), sim_mat.T)[:, :2]
        return dst, dst_lmks



    @staticmethod
    def piecewise_affine_transform(image, source_lmks, target_lmks):
        anchor = list(range(31)) + [36, 39, 42, 45, 48, 51, 54, 57]
        tgt_lmks = target_lmks[anchor, :]
        dst_lmks = source_lmks[anchor, :]
        tform = PiecewiseAffineTransform()
        tform.estimate(tgt_lmks, dst_lmks)
        dst = warp(image, tform, output_shape=image.shape[:2]).astype(np.float32)
        return dst



    def detect_faces(self, image):
        landmarks = self.face_detector(image)
        if len(landmarks) != 1:
            ValueError('Reference image had more than one face. I should only have one')

        landmark = landmarks[0]
        image_face, lmks = self.similarity_transform(image, landmark)
        
        return image_face, lmks
        


    def prep_image(self, image):
        """
        Runs images through the preprocessing steps
        :param image: A numpy image of shape (H, W, 3). The image should only have one face in it
        :return: Returns an image ready to be passed to the model
        """
        # Scaling the image to reduce its width to `scale_to`.
        # This makes sure that the run time is consistent by making sure the input image size is fixed.
        image = cv2.resize(image, (self.image_scale_to_before_crop, int(image.shape[0] * self.image_scale_to_before_crop/image.shape[1])), interpolation=cv2.INTER_AREA)
        # We need to `mean_lmks`, because `self.mean_lmks` is based on 240x320 resolution images
        mean_lmks = self.mean_lmks * self.image_scale_to_before_crop / 320

        # landmarks = self.face_detector(image)
        # if len(landmarks) > 1:
        #     ValueError('Reference image had more than one face. I should only have one')
        # else:
        #     landmark = landmarks[0]
        # image_face, lmks = self.similarity_transform(image, landmark)


        image_face, lmks = self.detect_faces(image)

        image_face = self.piecewise_affine_transform(image_face, lmks, mean_lmks)
        landmark = mean_lmks.round().astype(np.int)
        b_box = [landmark[:, 0].min(), landmark[:, 1].min(), landmark[:, 0].max(), landmark[:, 1].max()]
        image_face = self.crop_image(image_face, b_box)
        image_face = cv2.resize(image_face, (self.image_size, self.image_size))
        if len(image_face.shape) > 2 and image_face.shape[2] == 3:
            image_face = np.matmul(image_face, np.array([[0.114], [0.587], [0.299]]))
        image_face = self.clahe.apply((image_face * 255).astype(np.uint8))
        image_face = image_face.reshape(1, 1, self.image_size, self.image_size).astype(np.float32)
        return torch.from_numpy(image_face) / 255



    def test_image(self, image_path):
        image = cv2.imread(image_path)
        # Scaling the image to reduce its width to `scale_to`.
        # This makes sure that the run time is consistent by making sure the input image size is fixed.
        image = cv2.resize(image, (self.image_scale_to_before_crop, int(image.shape[0] * self.image_scale_to_before_crop/image.shape[1])), interpolation=cv2.INTER_AREA)

        try:
            self.detect_faces(image)
        except:
            return False

        return True

        

