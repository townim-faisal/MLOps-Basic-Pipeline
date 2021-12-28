import cv2
from skimage.metrics import structural_similarity as compare_ssim

def parallel_laplacian_variance(file):
    img = cv2.imread(file)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(img_gray, cv2.CV_64F).var()
    return laplacian

def compare_images(image1, image2):
    image_gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image_gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    try:
        diff, _ = compare_ssim(image_gray1, image_gray2, full=True)
    except:
        image_gray2 = cv2.resize(image_gray2, image_gray1.shape, interpolation = cv2.INTER_AREA)
        diff, _ = compare_ssim(image_gray1, image_gray2, full=True)
    return diff

def parallel_compare_images(i, files):
    image1 = cv2.imread(files[i])
    image2 = cv2.imread(files[i+1])
    image_gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image_gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    try:
        diff, _ = compare_ssim(image_gray1, image_gray2, full=True)
    except:
        image_gray2 = cv2.resize(image_gray2, (image_gray1.shape[1], image_gray1.shape[0]), interpolation = cv2.INTER_AREA)
        diff, _ = compare_ssim(image_gray1, image_gray2, full=True)
    # print(f'Similarity between {files[i]} and {files[i+1]}: {diff}', end="\r")
    return diff