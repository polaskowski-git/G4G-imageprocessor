#pylint: disable=E1101
#pylint: disable=missing-module-docstring
#pylint: disable=missing-class-docstring
#pylint: disable=missing-function-docstring
#pylint: disable-msg=too-many-locals
#pylint: disable-msg=too-many-statements

from datetime import datetime
import math
import base64
import numpy
import cv2

def process_image(img64, caliber, magazine_capacity, distance_to_target):

    # Funkcja zwracająca sformatowany znak czasu
    def return_timestamp():
        now = datetime.now()
        date_string = now.strftime('%Y-%m-%d-%H-%M')
        return date_string

    # Funkcja obliczająca odległość między dwoma punktami (pikselami)
    def calculate_distance(point1, point2):
        distance = math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
        return distance

    # Wczytanie zdjęcia z pliku tekstowego, konwersja do JPG oraz zmiana rozdzielczości
    def convert_base64_to_img(img64):
        base64_img_bytes = img64.encode('utf-8')

        with open('shield_raw_decoded.jpg', 'wb') as file_to_write:
            decoded_image_data = base64.decodebytes(base64_img_bytes)
            file_to_write.write(decoded_image_data)

        shield_raw = cv2.imread('shield_raw_decoded.jpg')
        img = shield_raw.copy()
        img = cv2.resize(img, (800, 800))
        return img

    # Oddzielenie pierwszego planu od tła (algorytm GrabCut)
    def separate_shield_from_background(img):
        mask = numpy.zeros(img.shape[:2], numpy.uint8)
        background_model = numpy.zeros((1, 65), numpy.float64)
        foreground_model = numpy.zeros((1, 65), numpy.float64)

        rectangle = (1, 1, 800, 800)

        cv2.grabCut(img, mask, rectangle, background_model,
                    foreground_model, 3, cv2.GC_INIT_WITH_RECT)

        mask2 = numpy.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        img = img * mask2[:, :, numpy.newaxis]
        return img

    # Wykrycie krawędzi i konturów tarczy oraz odpowiednie przycięcie zdjęcia
    def detect_contours_and_crop(img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.blur(img, (4, 4))

        canny_edged = cv2.Canny(img, 10, 250)
        contours, hier = cv2.findContours(canny_edged.copy(),
                                          cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hier = hier * 1
        for var_c in contours:
            var_x, var_y, var_w, var_h = cv2.boundingRect(var_c)
            if var_w > 400 and var_h > 400:
                img = img[var_y:var_y+var_h, var_x:var_x+var_w]
        return img

    def detect_shots(img):
        # Wykrycie dziur po kulach na tarczy (BlobDetector)
        params = cv2.SimpleBlobDetector_Params()
        params.filterByInertia = False
        params.filterByConvexity = False
        params.filterByColor = True
        params.blobColor = 0
        params.filterByArea = True
        params.minArea = 68 * caliber
        params.filterByCircularity = True
        params.minCircularity = 0.7

        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(img)

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        img = cv2.drawKeypoints(img, keypoints, numpy.array([]), (0, 0, 255),
                                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return img, keypoints

    # Wyznaczenie środka tarczy (układu współrzędnych) oraz promienia tarczy
    def define_shield_properties(img):
        height, width, channel = img.shape
        channel = channel * 1
        center = [int(height/2), int(width/2)]
        shield_radius = width/2
        return height, width, center, shield_radius

    # Określenie współrzędnych wykrytych dziur i liczenie punktów
    def calculate_coordinates_and_score(img, keypoints, center, shield_radius):
        shots_on_target = len(keypoints)
        shots = {}
        score = 0

        for keypoint in range(shots_on_target):
            var_y = int(keypoints[keypoint].pt[1])
            var_x = int(keypoints[keypoint].pt[0])
            shot = [var_y, var_x]
            shots[keypoint] = shot
            img[var_y, var_x] = [0, 255, 0]
            dist = calculate_distance(center, shot)
            if dist <= shield_radius * 0.1:
                score += 10
            elif dist <= shield_radius * 0.23:
                score += 9
            elif dist <= shield_radius * 0.36:
                score += 8
            elif dist <= shield_radius * 0.49:
                score += 7
            elif dist <= shield_radius * 0.62:
                score += 6
            elif dist <= shield_radius * 0.75:
                score += 5
            elif dist <= shield_radius * 0.88:
                score += 4
            elif dist <= shield_radius:
                score += 3
            else:
                score += 0

        return shots, score

    # Obliczenie pozostałych statystyk
    def calculate_shooting_statistics(shots, width):
        real_shield_diameter_length = 127
        max_group_size = 0

        accuracy = round((len(shots) / magazine_capacity)*100)

        for i in shots.values():
            for j in shots.values():
                group_size = calculate_distance(i, j)
                if group_size > max_group_size:
                    max_group_size = group_size

        max_group_size = round((max_group_size / width) * real_shield_diameter_length)
        shot_grouping = round(max_group_size / distance_to_target, 2)
        return accuracy, shot_grouping

    # Wykonanie programu
    img = convert_base64_to_img(img64)
    img = separate_shield_from_background(img)
    img = detect_contours_and_crop(img)
    img, keypoints = detect_shots(img)
    image_properties = define_shield_properties(img)
    shots, score = calculate_coordinates_and_score(img, keypoints,
                                                   image_properties[2], image_properties[3])
    accuracy, shot_grouping = calculate_shooting_statistics(shots, image_properties[1])



    cv2.imwrite('./output_img/image_{}.jpg'.format(return_timestamp()), img)
    return score, accuracy, shot_grouping
