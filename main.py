import cv2
import numpy

def augment_face(target: numpy.array, augment: numpy.array) -> numpy.array:
    target_height, target_width, _ = target.shape
    augment_height, augment_width, _ = augment.shape
    augment_copy = augment

    scalar = min(target_width / augment_width,
                target_height / augment_height)

    shape = (int(augment_width * scalar), int(augment_height * scalar))
    offset = (int((target_width - shape[0]) * 0.5), int((target_height - shape[1]) * 0.5))
    scaled_target = cv2.resize(augment, shape)


    augment_target = target.copy()
    #white_filter = (scaled_target < 250).all(axis=2)
     
    augment_target[offset[1]: offset[1] + shape[1], offset[0]: offset[0] + shape[0]] = scaled_target
    return augment_target


    
def main():
    augment = cv2.imread("bvarface/static/neutral.png", cv2.IMREAD_UNCHANGED)
    # image = augment[:,:,0:3]
    # augment_image = augment[:,:,3] / 255.0
    # augment_border = 1.0 - augment_image

    cap = cv2.VideoCapture(0)

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    while True:
        _, window = cap.read()
        bw = cv2.equalizeHist(cv2.cvtColor(window, cv2.COLOR_BGR2GRAY))

        faces = cascade.detectMultiScale(
            bw, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE)

        for x0, y0, x1, y1 in faces:
            width = (x0, x0 + x1)
            height = (int(y0 - (y1 * 0.25)), int(y0 + (y1 * 0.75)))

            if width[0] < 0 or width[1] > window.shape[1] or height[0] < 0 or height[1] > window.shape[0]:
                continue
        
            window[height[0]: height[1], width[0]: width[1]] = augment_face(window[height[0]: height[1], width[0]: width[1]], augment)
            #for c in range(0, 3):
            #    window[height[0]: height[1], width[0]: width[1], c] = \
            #        (image[:, :, c] * augment_image + window[height[0]: height[1], width[0]: width[1], c] * augment_border)

        cv2.imshow("BVAR Face", window)
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    
