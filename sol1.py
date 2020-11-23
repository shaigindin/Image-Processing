from skimage.color import rgb2gray
from imageio import imread
import matplotlib.pyplot as plt
import numpy as np



def read_image(filename, representation):
    """
    function read filename and return numpy array of the image
    :param filename: the filename of an image on disk (could be grayscale or RGB
    :param representation: representation code, either 1 or 2 defining whether the output should be a grayscale
           image (1) or an RGB image (2).
    :return:
    This function returns an image, make sure the output image is represented by a matrix of type
    np.float64 with intensities (either grayscale or RGB channel intensities) normalized to the range [0, 1].
    """
    try:
        im = imread(filename)
        im = im.astype(np.float64)
        im /= 255
        if (representation == 1):
            im = rgb2gray(im)
            return im
        elif (representation == 2):
            return im
        else:
            print("invalid input- representation out of range")
            exit(1)
    except:
        print("filep isnt valid")
        exit(1)


def imdisplay(filename, representation):
    """
    the function utilize read_image to display an image in a given representation
    :param filename: he filename of an image on disk (could be grayscale or RGB
    :param representation:  representation code, either 1 or 2 defining whether the output should be a grayscale
           image (1) or an RGB image (2).
    :return:
    """
    im = read_image(filename, representation)
    if (representation == 1):
        plt.imshow(im, cmap=plt.cm.gray)
        plt.show()
    elif (representation == 2):
        plt.imshow(im)
        plt.show()


def yiq2rgb(iq):
    """
    convert yiq to rgb
    :param iq: np.float64 matrice
    :return: np.float64 matrice
    """
    M = iq
    # the transpose of the inverse of the conversion matrice
    T = np.array([[1, 1, 1],
                  [242179 / 253408, -68821 / 253408, -280821 / 253408],
                  [157077 / 253408, -280821 / 253408, 432077 / 253408]])
    P = iq.reshape(M.shape[0] * M.shape[1], 3).dot(T)
    Result = P.reshape(M.shape[0], M.shape[1], 3)
    return Result


def rgb2yiq(imRGB):
    """
    convert rgb to yiq
    :param imRGB: np.float64 matrice
    :return: np.float64 matrice
    """
    M = imRGB

    # the transpose of the conversion matrice
    T = np.array([[0.299, 0.596, 0.212],
                  [0.587, -0.275, -0.523],
                  [0.114, -0.321, 0.311]])

    P = imRGB.reshape(M.shape[0] * M.shape[1], 3).dot(T)
    result = P.reshape(M.shape[0], M.shape[1], 3)
    return result


# delete in the end
def image_display(im, representation):
    if (representation == 1):
        plt.imshow(im, cmap=plt.cm.gray)
        plt.show()
    elif (representation == 2):
        plt.imshow(im)
        plt.show()


def get_histogram(img):
    """
    this function calculates histogram of img, and than calculates the cumsum histogram.
    than it normalize and strech the cumsum and we get the lookup table. we apply it on the image
    :param img: the image to equalized
    :return:
    """
    hist_orig, bin_edges = np.histogram(img.flatten(), bins=range(257))
    hist_cum = np.cumsum(hist_orig)
    c_255 = hist_cum[-1]
    c_m = hist_cum[(hist_cum != 0).argmax(axis=0)]
    lut = np.round(((hist_cum - c_m) / (c_255 - c_m)) * 255)
    im_eq = lut[img]
    hist_eq, bin_edges = np.histogram(im_eq.flatten(), bins=range(257))
    return im_eq, hist_orig, hist_eq


def change_pixels_scale_to_255(img):
    """
    scaling the np.float64 matrice to np.int matrice with values of [0,255]
    :param img: np.float64 matrice
    :return:
    """
    return np.round(img * 255).astype(np.int)


def histogram_equalize(img):
    """
    function that performs histogram equalization of a given grayscale or RGB image.
    :param img: np.float64 matrice
    :return:
    """
    # differentiate between grayscale and rgb
    if np.ndim(img) == 2:
        img = change_pixels_scale_to_255(img)
        im_eq, hist_orig, hist_eq = get_histogram(img)
        im_eq = im_eq.astype(np.float64)
        im_eq /= 255
        return [im_eq, hist_orig, hist_eq]
    elif np.ndim(img) == 3:
        iq = rgb2yiq(img)
        y = iq[:, :, 0]
        y = change_pixels_scale_to_255(y)
        im_eq, hist_orig, hist_eq = get_histogram(y)
        im_eq = im_eq.astype(np.float64)
        im_eq /= 255
        iq[:, :, 0] = im_eq
        im_eq = yiq2rgb(iq)
        return [im_eq, hist_orig, hist_eq]


def get_first_z(hist, number_of_pixels_in_quant):
    """
    initializing the borders in the beginning by calculating the same amount of pixels in each segment
    :param hist: histogram of the image
    :param number_of_pixels_in_quant: the derirable amount of pixels in each segment
    :return: the borders of the segments
    """
    THE_NUMBER_OF_GRAY_COLORS = 256
    z0 = -1
    z255 = 255
    z_current = z0
    z_list = []
    counter = 0
    for i in range(THE_NUMBER_OF_GRAY_COLORS):
        counter += hist[i]
        if counter >= number_of_pixels_in_quant:
            counter = 0
            z_list.append((np.int(z_current), np.int(i)))
            z_current = np.int(i)
    z_list.append((z_current, z255))
    z_list = check_validity(z_list)
    return z_list

def check_validity(z_list):
    z_start = z_list[-2][1]
    z_end = z_list[-1][1]
    if (z_end-z_start == 0):
        z_start = z_list[-2][0]
        z_end = z_list[-2][1]
        z_middle = np.round((z_end + z_start) / 2)
        z_list[-2] = (z_start,z_middle)
        z_list[-1] = (z_middle, 255)
    return z_list

def calculated_qi(z_i, z_start, z_end):
    """
    calculate the gray color for the segment
    :param z_i: the segment z_i
    :param z_start: the beginning color
    :param z_end: the ending color
    :return:  the q_i gray color
    """
    qi = 0
    counter = 0
    for h_g, g in zip(z_i, range(z_start, z_end)):
        qi += h_g * g
        counter += h_g
    return np.round(qi / counter)

#innter function for printing errors
def print_error(error_list):
    cur = error_list[0]
    for val in error_list[1:]:
        print(abs(cur - val))
        cur = val


def get_color(z_list, q_list, i):
    """
    calculating the color q_i from the calculated look-up table
    :param z_list: the final borders
    :param q_list: the final colors
    :param i: the current pixels we are looking the color for it
    :return:
    """
    counter = 0
    for x_start, x_end in z_list:
        if (x_start <= i <= x_end):
            return q_list[counter]
        counter += 1


def calculate_error(z_i, q_i, z_start, z_end):
    """

    :param z_i:
    :param q_i:
    :param z_start:
    :param z_end:
    :return:
    """
    error = 0
    for val, pixle_color in zip(z_i, range(z_start, z_end)):
        error += ((pixle_color - q_i) ** 2) * val
    return error


def is_diverged(e1, e2):
    """
    calculate the error difference
    :param e1: error 1
    :param e2: error2
    :return:
    """
    return (e1 - e2) == 0


def calculate_qlist_and_error(z_list, hist):
    """
    the function run over the entire segments and for each segment calculate the q_i
    :param z_list: list of tuples of segments each segment is a tuple of (start border,end border)
    :param hist: image histogram
    :return:
    """
    q_list = []
    error = 0
    for z_start, z_end in z_list:
        z_i = hist[z_start.__int__() + 1: z_end.__int__()]
        q_i = calculated_qi(z_i, z_start.__int__() + 1, z_end.__int__())
        q_list.append(q_i)
        error += calculate_error(z_i, q_i, z_start.__int__() + 1, z_end.__int__())
    return q_list, error


def calculate_zlist(z_list, q_list):
    """
    function calculate for each q_i new border
    :param z_list:
    :param q_list:
    :return:
    """
    for i in range(0, len(z_list) - 1):
        z_i = np.round((q_list[i] + q_list[i + 1]) / 2)
        z_list[i] = (z_list[i][0], z_i)
        z_list[i + 1] = (z_i, z_list[i + 1][1])
    return z_list


def calculate_number_of_pixels_in_quant(hist, n_quant):
    """
    calculates the apropriate number of pixels in each segment in the beginning (should be equal)
    :param hist: the image histogram
    :param n_quant: number of quants
    :return:
    """
    hist_cum = np.cumsum(hist)
    number_of_pixels = hist_cum[-1]
    number_of_pixls_in_quant = number_of_pixels / n_quant
    return number_of_pixls_in_quant


def quantize_helper(im_orig, n_quant, n_iter):
    """
    this function is the actual quantize algorithm
    :param im_orig: is the input grayscale or RGB image to be quantized (float64 image with values in [0, 1]).
    :param n_quant: is the number of intensities your output im_quant image sho
    :param n_iter:  is the maximum number of iterations of the optimization procedure (may converge earlier.)
                    And the output is a list [im_quant, error] whe
    :return:
    """
    img = change_pixels_scale_to_255(im_orig)
    hist, bin_edges = np.histogram(img.flatten(), bins=range(257))
    number_of_pixels_in_quant = calculate_number_of_pixels_in_quant(hist, n_quant)
    z_list = get_first_z(hist, number_of_pixels_in_quant)
    error_list = []

    #check input validity
    if (n_iter < 1):
        print("n_iter should be at list one")
        exit(1)

    for i in range(n_iter):
        q_list, error = calculate_qlist_and_error(z_list, hist)
        z_list = calculate_zlist(z_list, q_list)

        if len(error_list) >= 1:
            if (is_diverged(error_list[-1], error)):
                break
        error_list.append(error)

    lut = []

    for i in range(256):
        lut.append(get_color(z_list, q_list, i))
    lut = np.array(lut)

    new_img = lut[img]

    new_img = new_img.astype(np.float64)
    new_img /= 255
    return new_img, error_list


def quantize(im_orig, n_quant, n_iter):
    """
    function that performs optimal quantization of a given grayscale or RGB image
    :param im_orig: is the input grayscale or RGB image to be quantized (float64 image with values in [0, 1]).
    :param n_quant: is the number of intensities your output im_quant image sho
    :param n_iter:  is the maximum number of iterations of the optimization procedure (may converge earlier.)
                    And the output is a list [im_quant, error] where
    :return:
    im_quant -  is the quantized output image. (float64 image with values in [0, 1]).
    error    -  is an array with shape (n_iter,) (or less) of the total intensities error for each iteration of the
                quantization procedure.
    """
    if np.ndim(im_orig) == 2:
        im_quant, error = quantize_helper(im_orig, n_quant, n_iter)
        return im_quant, error
    elif np.ndim(im_orig) == 3:
        iq = rgb2yiq(im_orig)
        y = iq[:, :, 0]
        im_eq, error = quantize_helper(y, n_quant, n_iter)
        iq[:, :, 0] = im_eq
        im_quant = yiq2rgb(iq)
        return im_quant, error

# path = "/Users/shaigindin/temp/2.jpeg"

#
# im = read_image(path, 2)
# iq = rgb2yiq(im)
# iq_true = iqi(im)
# print(iq)
# print("****")
# print(iq_true)
# iq = np.clip(iq,0,1)
# image_display(iq,2)
# im = yiq2rgb(iq)
# im = np.clip(im,0,1)
# image_display(im,2)

# x = np.hstack([np.repeat(np.arange(1,51,2),10)[None,:], np.array([255]*6)[None,:]])
# grad = np.tile(x,(256,1))
# grad = grad.astype(np.float64)
# grad /= 255
#
# image_display(grad, 1)
#
# im_gray = read_image(path, 1)
# image_display(im_gray,1)
#
# im_color = read_image(path, 2)
# image_display(im_color, 2)
# t,a,x = histogram_equalize(im_gray)
# t = np.clip(t, 0, 1)
# image_display(t ,1)
# t, error = quantize(im_color, 20, 3)
#
# t = np.clip(t, 0, 1)
# image_display(t ,1)

# print(error)
# imdisplay(path,1)
# plt.imshow(sk.color.yiq2rgb(sk.color.rgb2yiq(im)))
# plt.show()
