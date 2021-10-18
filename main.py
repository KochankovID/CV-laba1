import cv2
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
    f, ax = plt.subplots(5, 2, figsize=(4, 10))

    face_cascade = cv2.CascadeClassifier("trained_face_recognition.xml")
    img = cv2.imread("test_img.png")

    ax[0][0].set_title('Actual image')
    ax[0][0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax[0][0].axis('off')

    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img_grey, 1.1, 1)

    for (x, y, w, h) in faces:
        cropped_img = img[(y - int(y * 0.1)): (y + int(h * 1.1)), (x - int(x * 0.1)): (x + int(w * 1.1))].copy()

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    ax[0][1].set_title('Find face')
    ax[0][1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax[0][1].axis('off')

    ax[1][0].set_title('Cropped image')
    ax[1][0].imshow(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
    ax[1][0].axis('off')

    cropped_img_grey = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    img_binary = cv2.Canny(cropped_img_grey, 100, 100)

    ax[1][1].set_title('Binary edges')
    ax[1][1].imshow(cv2.cvtColor(img_binary, cv2.COLOR_BGR2RGB))
    ax[1][1].axis('off')

    nb_components, labels, stats, centroids = cv2.connectedComponentsWithStats(img_binary, 4)
    mask = np.zeros(cropped_img_grey.shape, dtype="uint8")

    for i in range(1, nb_components):
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        if w < 10 and h < 10:
            continue

        componentMask = (labels == i).astype("uint8") * 255
        mask = cv2.bitwise_or(mask, componentMask)

    ax[2][0].set_title('Binary edges (filtered)')
    ax[2][0].imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
    ax[2][0].axis('off')

    kernel = np.ones((5, 5), "uint8")
    img_dilated = cv2.dilate(img_binary, kernel, iterations=1)

    ax[2][1].set_title('Dilation')
    ax[2][1].imshow(cv2.cvtColor(img_dilated, cv2.COLOR_BGR2RGB))
    ax[2][1].axis('off')

    M: np.ndarray = cv2.GaussianBlur(img_dilated, ksize=(5, 5), sigmaX=5, sigmaY=5).astype(np.float32)
    M /= 255

    ax[3][0].set_title('M')
    ax[3][0].imshow(cv2.cvtColor(M, cv2.COLOR_BGR2RGB))
    ax[3][0].axis('off')

    F1: np.ndarray = cv2.bilateralFilter(cropped_img, 5, 100, 100)
    ax[3][1].set_title('F1')
    ax[3][1].imshow(cv2.cvtColor(F1, cv2.COLOR_BGR2RGB))
    ax[3][1].axis('off')

    F2: np.ndarray = cv2.addWeighted(cropped_img, 2.6, F1, -1.5, 0)
    ax[4][0].set_title('F2')
    ax[4][0].imshow(cv2.cvtColor(F2, cv2.COLOR_BGR2RGB))
    ax[4][0].axis('off')

    final = M[:, :, np.newaxis].repeat(3, 2) * F2 + (1 - M)[:, :, np.newaxis] * F1
    final = final.astype(np.uint8)

    ax[4][1].set_title('final')
    ax[4][1].imshow(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
    ax[4][1].axis('off')

    f.tight_layout()
    plt.show()
