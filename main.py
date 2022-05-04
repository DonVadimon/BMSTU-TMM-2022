import enum

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

        cv2.createTrackbar(
            FacePartsRanges.LEFT_BROW.name, self.NAME, 0, 1, self.on_left_brow_change)
        cv2.createTrackbar(
            FacePartsRanges.RIGHT_BROW.name, self.NAME, 0, 1, self.on_right_brow_change)

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

    def on_left_brow_change(self, value):
        self.on_face_part_option_change(FacePartsRanges.LEFT_BROW.name, value)

    def on_right_brow_change(self, value):
        self.on_face_part_option_change(FacePartsRanges.RIGHT_BROW.name, value)

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
    # FACE_SHAPE = slice(0, 17)
    LEFT_BROW = slice(17, 22)
    RIGHT_BROW = slice(22, 27)
    # NOSE = slice(27, 36)
    LEFT_EYE = slice(36, 42)
    RIGHT_EYE = slice(42, 48)
    LIPS = slice(48, 69)


class ImageConverter:
    INITIAL_SCALE = 1

    def __init__(self, img):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(
            'shape_predictor_68_face_landmarks.dat')

        self.img = img.copy()
        self.img_original = img.copy()
        self.img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    def processFace(self, face, face_parts,  red, green, blue):
        landmarks = self.predictor(self.img_gray, face)
        points = []

        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            points.append([x, y])

        points = np.array(points)

        processed_parts = list()
        bw_mask = np.zeros_like(self.img_original)

        for part in FacePartsRanges.__members__:
            part_range = FacePartsRanges[part].value
            part_bw_mask = self.create_bw_mask_by_points(
                self.img_original, points[part_range])

            cv2.imshow('part_bw_mask', part_bw_mask)

            bw_mask = cv2.add(bw_mask, part_bw_mask)

            if part_range in face_parts:

                merged_img = self.img_original.copy()

                color_mask = np.zeros_like(part_bw_mask)
                color_mask[:] = red, green, blue

                cv2.imshow('color_mask', color_mask)

                color_mask = cv2.bitwise_and(part_bw_mask, color_mask)
                color_mask = cv2.GaussianBlur(color_mask, (7, 7), 10)

                merged_img = cv2.addWeighted(merged_img, 1, color_mask, 0.4, 0)

                processed_part = self.crop_by_points(
                    merged_img, points[part_range])

                processed_parts.append(processed_part.copy())
            else:
                processed_part = self.crop_by_points(
                    self.img, points[part_range])
                cv2.imshow('processed_part', processed_part)
                processed_parts.append(processed_part.copy())

        merged_mask = np.zeros_like(self.img_original)

        for processed_part in processed_parts:
            merged_mask = cv2.addWeighted(
                merged_mask, 1, processed_part, 1, 0)

        cv2.imshow('merged_mask', merged_mask)

        cv2.imshow('bw_mask', bw_mask)

        bw_mask_inversed = cv2.bitwise_not(bw_mask)
        cv2.imshow('bw_mask_inversed', bw_mask_inversed)

        bw_mask_merged = cv2.bitwise_and(bw_mask_inversed, self.img)

        cv2.imshow('bw_mask_merged', bw_mask_merged)

        proc_img = cv2.add(bw_mask_merged, merged_mask)
        cv2.imshow('proc_img', proc_img)

        return proc_img

    def processImage(self, face_parts, red, green, blue):
        faces = self.detector(self.img_gray)

        processed_images = []

        for face in faces:
            proc_img = self.processFace(
                face, face_parts, red, green, blue)

            # merged_img = cv2.bitwise_not(merged_mask)

            # merged_img = cv2.addWeighted(
            #     self.img_original, 1, merged_mask, 0, 0)

            processed_images.append(proc_img)

            self.img = proc_img

        return processed_images

    def crop_by_points(self, img, points):
        mask = np.zeros(self.img_original.shape[:2], dtype="uint8")
        mask = cv2.fillPoly(
            mask, [points], (255, 255, 255))
        return cv2.bitwise_and(img, img, mask=mask)

    def create_bw_mask_by_points(self, img, points):
        mask = np.zeros_like(img)
        mask = cv2.fillPoly(mask, [points], (255, 255, 255))
        img = cv2.bitwise_and(img, mask)
        return mask


AppWindow('./toxa.jpeg')
