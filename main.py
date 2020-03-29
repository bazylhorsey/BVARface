import cv2

def main():
    cap = cv2.VideoCapture(0)

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    while True:
        _, window = cap.read()

        bw = cv2.equalizeHist(cv2.cvtColor(window, cv2.COLOR_BGR2GRAY))

        faces = cascade.detectMultiScale(
            bw, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE)

        for x1, y1, x2, y2 in faces:
            cv2.rectangle(window, (x1, y1), (x1 + x2, y1 + y2), (0, 255, 0), 2)

        cv2.imshow("BVAR Face", window)
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
