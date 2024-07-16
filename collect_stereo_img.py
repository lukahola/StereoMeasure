import cv2
import os

cap_1 = cv2.VideoCapture(0)
cap_2 = cv2.VideoCapture(2)
num = 0
while cap_1.isOpened() and cap_2.isOpened():
    
    img_path = "F:/Projects/pythonProject/stereo/200mm/"
    left_img_path = img_path + "left/"
    right_img_path = img_path + "right/"
    if not os.path.exists(left_img_path):
        os.mkdir(left_img_path)
    if not os.path.exists(right_img_path):
        os.mkdir(right_img_path)

    ret1, frame1 = cap_1.read()
    ret2, frame2 = cap_2.read()
    cv2.namedWindow("left")
    cv2.namedWindow("right")

    if not ret1 or not ret2:
        KeyError("No input!")
        break

    else:
        cv2.imshow("left", frame1)
        cv2.imshow("right", frame2)

        if cv2.waitKey(1) == ord("s"):
            num += 1

            left_name = left_img_path + str(num) + ".jpg"
            print("Save to " + left_name)
            right_name = right_img_path + str(num) + ".jpg"
            cv2.imwrite(left_name, frame1)
            cv2.imwrite(right_name, frame2)
            continue

        elif cv2.waitKey(1) == ord("q"):
            cap_1.release
            cap_2.release
            cv2.destroyAllWindows()
