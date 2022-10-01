import numpy as np

from getImages import *

home = "../input/"
sobel_kernel = np.array(([-1, 0, 1], [-2, 0, 2], [-1, 0, 1]))

def generateGaussian(size, scaleX, scaleY):
    lower_limit = int(-((size - 1) / 2))
    upper_limit = abs(lower_limit) + 1
    ind = np.arange(lower_limit, upper_limit)
    row = np.reshape(ind, (ind.shape[0], 1)) + np.zeros((1, ind.shape[0]))
    col = np.reshape(ind, (1, ind.shape[0])) + np.zeros((ind.shape[0], 1))
    G = (1 / (2 * np.pi * (scaleX * scaleY))) * np.exp(
        -(((col) ** 2 / (2 * (scaleX ** 2))) + ((row) ** 2 / (2 * (scaleY ** 2)))))
    return G


def Gblur(size, sig, image):
    kern = generateGaussian(size, sig, sig)
    blur = cv2.filter2D(image, -1, kern)
    return blur


def rotateDoG(DoG, angle):
    h, w = DoG.shape
    cX, cY = w // 2, h // 2
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    rotatedDoG = cv2.warpAffine(DoG, M, (w, h))
    return rotatedDoG


def generateDerivatives(gaussians, kernel):
    first_order = []
    second_order = []
    x = kernel
    y = kernel.T
    for gaussian in gaussians:
        DoGx = cv2.filter2D(gaussian, -1, x)
        DoGy = cv2.filter2D(gaussian, -1, y)
        DoG = (0.5 * DoGx) + (0.5 * DoGy)
        first_order.append(DoG)
    for first_derivative in first_order:
        DoGx = cv2.filter2D(first_derivative, -1, x)
        DoGy = cv2.filter2D(first_derivative, -1, y)
        DoG = (0.5 * DoGx) + (0.5 * DoGy)
        second_order.append(DoG)
    return first_order, second_order


def generateRotations(first_derivative, second_derivative, rotations):
    first_derivative_rot = []
    second_derivative_rot = []
    for i in range(len(first_derivative)):
        for j in range(len(rotations)):
            first_derivative_rot.append(rotateDoG(first_derivative[i], rotations[j]))
            second_derivative_rot.append(rotateDoG(second_derivative[i], rotations[j]))
    return second_derivative_rot

def generateLM(size, kernel):
    rotations = [0, 30, 60, 90, 120, 150]
    scales = [1, 2 ** (1 / 2), 2, 2 * (2 ** (1 / 2))]
    gaussians_for_DoG = []
    LMScales = []
    [LMScales.append([i, 3 * i]) for i in scales]
    for i in range(len(LMScales)):
        gaussians_for_DoG.append(generateGaussian(size, LMScales[i][0], LMScales[i][1]))

    first_derivative, second_derivative = generateDerivatives(gaussians_for_DoG[:3], kernel)
    rotations_2D = generateRotations(first_derivative, second_derivative, rotations)
    return rotations_2D[:5]



def getMagTh(grad_x, grad_y):
    np.seterr(divide='ignore', invalid='ignore')
    new_x = np.amax(grad_x, axis=2)
    new_y = np.amax(grad_y, axis=2)
    img_mag = np.sqrt(np.power(new_x, 2) + np.power(new_y, 2))
    img_th = np.arctan(new_y / new_x)
    return img_mag.astype(np.single), img_th.astype(np.single)


def nonMax(image, mag):
    image = Gblur(3, 1, image)
    low = 50
    high = 150
    edges = cv2.Canny(image, low, high)
    mag_temp = mag.copy()
    for i in range(mag_temp.shape[0]):
        for j in range(mag_temp.shape[1]):
            if edges[i, j] <= 0:
                mag_temp[i, j] = 0
    return mag_temp


def getOrientEdges(mag, filters, image, thresh):
    mag = Gblur(3, 1, mag)
    list_mag = []
    for i in range(len(filters)):
        temp = cv2.filter2D(mag, -1, filters[i])
        list_mag.append(temp)
    l = list_mag[0].copy()
    for j in range(1, len(list_mag)):
        l = (l - list_mag[j]) + list_mag[j]

    l = threshold(nonMax(image, rescale(l)), thresh)
    l = (l * 255).astype(np.uint8)
    return l

def rescale(array):
    array = (array - np.min(array)) / (np.max(array) - np.min(array))
    array = array / np.max(array)
    return array


def getGradientEdges(image, kernel, thresh):
    y1 = kernel
    x1 = y1.T
    y2 = kernel * (-1)
    x2 = y2.T
    im_x1 = cv2.filter2D(Gblur(3, 1, image), -1, x1)
    im_y1 = cv2.filter2D(Gblur(3, 1, image), -1, y1)
    im_x2 = cv2.filter2D(Gblur(3, 1, image), -1, x2)
    im_y2 = cv2.filter2D(Gblur(3, 1, image), -1, y2)
    mag1, orient1 = getMagTh(im_x1, im_y1)
    mag2, orient2 = getMagTh(im_x2, im_y2)
    mag_out = mag2 + (mag2 - mag1)
    edge = threshold(nonMax(image, rescale(mag_out)), thresh)
    edge = (edge * 255).astype(np.uint8)

    return edge, mag_out

def threshold(image, thresh):
    image[image < thresh] = 0
    image[image >= thresh] = 1
    return image

save = "../output/"
list_images = getImages(home)
list_filters = generateLM(5, sobel_kernel)

for i in range(len(list_images)):
    grad_edges, grad_mag = getGradientEdges(list_images[i], sobel_kernel, 0.05)
    orient_edges = getOrientEdges(grad_mag, list_filters, list_images[i], 0.05)
    cv2.imshow("Original", list_images[i])
    cv2.imshow("Gradient Edges", grad_edges)
    cv2.imshow("Orientation Edges", orient_edges)
    save_dir = save + "/Image_" + str(i + 1) + "_"
    cv2.imwrite(save_dir + "Gradient.png", grad_edges)
    cv2.imwrite(save_dir + "Orientation.png", orient_edges)
    cv2.waitKey(0)
