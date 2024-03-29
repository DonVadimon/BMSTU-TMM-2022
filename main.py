import enum

# ? импортируем необходимые библиотеки
import enum
import cv2
import numpy as np
import dlib

# ? класс окна приложения.
# ? используется для отрисовки интерфейса и вывода изображения
class AppWindow():
    # ? параметры окна
    WIDTH = 240
    HEIGHT = 640
    NAME = 'SugarLips'

    def __init__(self, img=None, imgPath=None, waitKey=0, video=False):
        self.video = video

        img_to_process = None

        # ? выбор источника изображения
        if imgPath is not None:
            img_to_process = cv2.imread(imgPath)
        if img is not None:
            img_to_process = img

        if img_to_process is None:
            raise Exception('No image provided')

        self.img_to_process = img_to_process

        self.converter = ImageConverter(img_to_process)

        # ? задание изначальных значений цвета макияжа
        self.red = 0
        self.green = 0
        self.blue = 0
        self.selected_parts = set()

        # ? создание элементов интерфейса
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

        # ? вывод изначального изображения
        cv2.imshow(self.NAME, img_to_process)

        if waitKey is not None:
            cv2.waitKey(waitKey)

    # ? метод для отрисовки изображения
    def show_image(self, img):
        cv2.imshow(self.NAME, img)

    # ? метод для смены обработки изображения
    def update_window(self, img=None):
        # ? обновление обрабатываемого изображения
        if img is not None:
            self.img_to_process = img
            self.converter.update_all_images(self.img_to_process)


        # ? список обработанных частей лица
        processing_parts = []
        for part_name in self.selected_parts:
            processing_parts.append(FacePartsRanges[part_name].value)

        # ? список обработанных изображений (по одному на лицо)
        processed_images = self.converter.process_image(
            face_parts=processing_parts,
            red=self.red, green=self.green, blue=self.blue)

        for image in processed_images:
            self.show_image(image)

    # ? обработчик пользовательского ввода выбора части лица для обработки
    def on_face_part_option_change(self, part_name, value):
        if part_name in self.selected_parts and value == 0:
            self.selected_parts.remove(
                part_name)
        if not part_name in self.selected_parts and value == 1:
            self.selected_parts.add(part_name)

    # ? обработчики конкретных частей
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

    # ? обработчики пользовательского ввода выбора цвета макияжа
    def on_red_trackbar_change(self, value):
        self.red = value
        if not self.video:
            self.update_window()

    def on_green_trackbar_change(self, value):
        self.green = value
        if not self.video:
            self.update_window()

    def on_blue_trackbar_change(self, value):
        self.blue = value
        if not self.video:
            self.update_window()


# ? отображение частей лица и номеров точек с их очертаниями
class FacePartsRanges(enum.Enum):
    # FACE_SHAPE = slice(0, 17)
    LEFT_BROW = slice(17, 22)
    RIGHT_BROW = slice(22, 27)
    # NOSE = slice(27, 36)
    LEFT_EYE = slice(36, 42)
    RIGHT_EYE = slice(42, 48)
    LIPS = slice(48, 69)


# ? класс для обработки изобрадений
class ImageConverter:
    INITIAL_SCALE = 1

    def __init__(self, img):
        # ? создание инстансов моделей для распознавания лиц
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(
            'shape_predictor_68_face_landmarks.dat')

        self.img = img.copy()
        self.img_original = img.copy()

    # ? метод для обработки лица
    def process_face(self, face, face_parts, img_gray, red, green, blue):
        # ? поиск точек с очертаниями частей лица
        landmarks = self.predictor(img_gray, face)
        points = []

        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            points.append([x, y])

        points = np.array(points)

        processed_parts = list()
        # ? создание черного изображения по размеры оригинального
        bw_mask = np.zeros_like(self.img_original)

        for part in FacePartsRanges.__members__:
            part_range = FacePartsRanges[part].value
            # ? создание черно-белой маски для обрабатываемой части лица
            part_bw_mask = self.create_bw_mask_by_points(
                self.img_original, points[part_range])

            # ? дополнение полной маски
            bw_mask = cv2.add(bw_mask, part_bw_mask)

            if part_range in face_parts:

                merged_img = self.img_original.copy()

                # ? создание цветовой маски
                color_mask = np.zeros_like(part_bw_mask)
                color_mask[:] = red, green, blue
                # ? наложение цветовой маски на обрабатываемуб часть
                color_mask = cv2.bitwise_and(part_bw_mask, color_mask)
                # ? добавление Гауссовского размытия для придания пастельного оттенка
                color_mask = cv2.GaussianBlur(color_mask, (7, 7), 10)

                # ? наложение цветовой маски на изначальное изображение
                merged_img = cv2.addWeighted(merged_img, 1, color_mask, 0.4, 0)

                # ? вырезание обработанной части
                processed_part = self.crop_by_points(
                    merged_img, points[part_range])

                processed_parts.append(processed_part.copy())
            else:
                # ? обрезание изображения спредыдущей обработки, чтобы не изменять его
                processed_part = self.crop_by_points(
                    self.img, points[part_range])
                processed_parts.append(processed_part.copy())

        merged_mask = np.zeros_like(self.img_original)

        for processed_part in processed_parts:
            merged_mask = cv2.addWeighted(
                merged_mask, 1, processed_part, 1, 0)

        # ? инвертирование маски
        bw_mask_inversed = cv2.bitwise_not(bw_mask)

        # ? вырезание обработанных частей
        bw_mask_merged = cv2.bitwise_and(bw_mask_inversed, self.img)

        # ? совмещение обработанной маски и обрезанного изображения
        proc_img = cv2.add(bw_mask_merged, merged_mask)

        return proc_img

    # ? метод для обработки изображения
    def process_image(self, face_parts, red, green, blue):
        # ? приведение изображения из трехканальног к двухканальному (черно-белому)
        img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        # ? обнаружение лиц
        faces = self.detector(img_gray)

        processed_images = []

        # ? обработка каждого лица
        for face in faces:
            proc_img = self.process_face(
                face, face_parts, img_gray, red, green, blue)

            processed_images.append(proc_img)

            self.img = proc_img

        return processed_images

    # ? метод для обрезки изображения по точкам
    def crop_by_points(self, img, points):
        mask = np.zeros(self.img_original.shape[:2], dtype="uint8")
        mask = cv2.fillPoly(
            mask, [points], (255, 255, 255))
        return cv2.bitwise_and(img, img, mask=mask)

    # ? метод для создания черно-белой маски по точкам
    def create_bw_mask_by_points(self, img, points):
        mask = np.zeros_like(img)
        mask = cv2.fillPoly(mask, [points], (255, 255, 255))
        img = cv2.bitwise_and(img, mask)
        return mask

    def update_all_images(self, img_original):
        self.img_original = img_original
        self.img = img_original


# ? функци ядл запуска изображения с использованием камеры
def run_with_video():
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)

    rval, frame = vc.read()
    app_win = None

    while True:

        if frame is not None:
            if app_win is None:
                app_win = AppWindow(img=frame, waitKey=None, video=True)
            else:
                app_win.update_window(img=frame)

        rval, frame = vc.read()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


run_with_video()

# img = cv2.imread('./face.jpeg')

# AppWindow(img=img)
