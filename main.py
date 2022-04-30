import enum
import imp

import enum
from os import remove
import cv2
import numpy as np
import dlib


class AppWindow():
    WIDTH = 240
    HEIGHT = 640
    NAME = 'SugarLips'

    def __init__(self, imgPath):
        img = cv2.imread(imgPath)
        self.converter = ImageConverter(img)

        self.red = 0
        self.green = 0
        self.blue = 0
        self.selected_parts = set()

        cv2.namedWindow(self.NAME)
        cv2.resizeWindow(self.NAME, self.WIDTH, self.HEIGHT)
        cv2.createTrackbar('Red', self.NAME, 0, 255,
                           self.on_red_trackbar_change)
        cv2.createTrackbar('Green', self.NAME, 0, 255,
                           self.on_green_trackbar_change)
        cv2.createTrackbar('Blue', self.NAME, 0, 255,
                           self.on_blue_trackbar_change)

        cv2.createTrackbar(
            FacePartsRanges.LIPS.name, self.NAME, 0, 1, self.on_lips_change)
        cv2.createTrackbar(
            FacePartsRanges.LEFT_EYE.name, self.NAME, 0, 1, self.on_left_eye_change)
        cv2.createTrackbar(
            FacePartsRanges.RIGHT_EYE.name, self.NAME, 0, 1, self.on_right_eye_change)

        cv2.imshow(self.NAME, img)
        cv2.waitKey(0)

    def show_image(self, img):
        cv2.imshow(self.NAME, img)

    def update_window(self):
        processing_parts = []
        for part_name in self.selected_parts:
            processing_parts.append(FacePartsRanges[part_name].value)

        processed_images = self.converter.processImage(
            face_parts=processing_parts,
            red=self.red, green=self.green, blue=self.blue)

        for image in processed_images:
            self.show_image(image)

    def on_face_part_option_change(self, part_name, value):
        if part_name in self.selected_parts and value == 0:
            self.selected_parts.remove(
                part_name)
        if not part_name in self.selected_parts and value == 1:
            self.selected_parts.add(part_name)

    def on_lips_change(self, value):
        self.on_face_part_option_change(FacePartsRanges.LIPS.name, value)

    def on_left_eye_change(self, value):
        self.on_face_part_option_change(FacePartsRanges.LEFT_EYE.name, value)

    def on_right_eye_change(self, value):
        self.on_face_part_option_change(FacePartsRanges.RIGHT_EYE.name, value)

    def on_red_trackbar_change(self, value):
        self.red = value
        self.update_window()

    def on_green_trackbar_change(self, value):
        self.green = value
        self.update_window()

    def on_blue_trackbar_change(self, value):
        self.blue = value
        self.update_window()


class FacePartsRanges(enum.Enum):
    FACE_SHAPE = slice(0, 17)
    LEFT_BROW = slice(17, 22)
    RIGHT_BROW = slice(22, 27)
    NOSE = slice(27, 36)
    LEFT_EYE = slice(36, 42)
    RIGHT_EYE = slice(42, 48)
    LIPS = slice(48, 69)


class ImageConverter:
    INITIAL_SCALE = 1

    def __init__(self, img):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(
            'shape_predictor_68_face_landmarks.dat')

        self.img = cv2.resize(
            img, (0, 0), None, self.INITIAL_SCALE, self.INITIAL_SCALE)
        self.img_original = img.copy()
        self.imgGray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        # ! here we go
        self.changed_parts = set()

    def processFace(self, face, face_parts,  red, green, blue):
        landmarks = self.predictor(self.imgGray, face)
        points = []

        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            points.append([x, y])

        merged_img = self.img_original.copy()
        points = np.array(points)

        for part in FacePartsRanges.__members__:
            if FacePartsRanges[part].value in face_parts:
                part_box = self.createBox(
                    self.img_original, points[FacePartsRanges[part].value], masked=True, cropped=False)

                color_mask = np.zeros_like(part_box)
                color_mask[:] = red, green, blue
                color_mask = cv2.bitwise_and(part_box, color_mask)
                color_mask = cv2.GaussianBlur(color_mask, (7, 7), 10)

                merged_img = cv2.addWeighted(merged_img, 1, color_mask, 0.4, 0)
                self.img = merged_img
            else:
                mask = np.zeros(self.img_original.shape[:2], dtype="uint8")
                mask = cv2.fillPoly(
                    mask, [points[FacePartsRanges[part].value]], (255, 255, 255))
                # cv2.imshow('mask', mask)

                prev_colored_part_box = cv2.bitwise_and(
                    self.img, self.img, mask=mask)

                # cv2.imshow('masked', prev_colored_part_box)

                merged_img = cv2.addWeighted(
                    merged_img, 1, prev_colored_part_box, 0, 0)

        return merged_img

    def processImage(self, face_parts, red, green, blue):
        faces = self.detector(self.imgGray)

        processed_images = []

        for face in faces:
            processed_images.append(self.processFace(
                face, face_parts, red, green, blue))

        return processed_images

    def createBox(self, img, points, scale=1, masked=False, cropped=True):
        if masked:
            mask = np.zeros_like(img)
            mask = cv2.fillPoly(mask, [points], (255, 255, 255))
            img = cv2.bitwise_and(img, mask)
            return mask
        if cropped:
            bbox = cv2.boundingRect(points)
            x, y, w, h = bbox
            img_crop = img[y: y + h, x: x + w]
            img_crop = cv2.resize(img_crop, (0, 0), None, scale, scale)
            return img_crop


AppWindow('./toxa.jpeg')
