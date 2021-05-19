import argparse
import numpy as np
from numpy.linalg.linalg import _raise_linalgerror_svd_nonconvergence
from skimage import io
from skimage import img_as_float, img_as_uint
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
# parser.add_argument("-v", "--verbose", help="increase output verbosity")
parser.add_argument("-f", "--file", help="plik z oryginalnym obrazem")
parser.add_argument("-out", "--output_file", help="nazwa pliku wyjściowego")
parser.add_argument(
    "-svd",
    help="implementacja SVD do użycia",
    choices=["custom", "library"],
    default="custom",
)
parser.add_argument(
    "-k", type=int, help="liczba wartości osobliwych użyta do kompresji"
)
args = parser.parse_args()


def library_svd(img, k):
    shape = img.shape
    image_reshaped = img.reshape((shape[0], shape[1] * 3))

    u, s, v = np.linalg.svd(image_reshaped, full_matrices=False)
    compressed = np.dot(u[:, :k], np.dot(np.diag(s[:k]), v[:k, :]))
    compressed_reshaped = compressed.reshape(shape)

    # print("u.shape: ", u.shape)
    # print("s.shape: ", s.shape)
    # print("v.shape: ", v.shape)

    full_variantion = np.diag(s).sum()
    selected_variantion = np.diag(s[:k]).sum()
    variantion_ratio = selected_variantion / full_variantion
    # print(f"Compression {100-variantion_ratio*100}%")

    fig, axes = plt.subplots(1, 2, figsize=(10, 7))
    ax = axes.ravel()
    ax[0].imshow(img)
    ax[0].set_title("Original")
    ax[1].imshow(compressed_reshaped)
    ax[1].set_title(
        f"Compression: {np.around(100-variantion_ratio*100)}% {k}/{s.shape[0]} eigenvalues"
    )
    fig.tight_layout()
    if args.output_file:
        plt.savefig(args.output_file)
    else:
        plt.show()


def add_padding(arr, n, m):
    res = arr.copy()
    if n > arr.shape[0]:
        diff = n - arr.shape[0]
        res = np.pad(res, ((0, diff), (0, 0)), "constant", constant_values=0)
    if m > arr.shape[1]:
        diff = m - arr.shape[1]
        res = np.pad(res, ((0, 0), (0, diff)), "constant", constant_values=0)

    return res[:n, :m]


def custom_svd(img, k):
    shape = img.shape
    # print("shape: ", shape)
    image_reshaped = img.reshape((shape[0], shape[1] * 3))
    # print("reshaped: ", image_reshaped.shape)

    # U, S, V = np.linalg.svd(image_reshaped, full_matrices=False)

    ### V
    imgT_img = np.dot(image_reshaped.T, image_reshaped)
    eigenvaluesV, eigenvectorsV = np.linalg.eigh(imgT_img)
    eigenvaluesV_sorted_ids = np.argsort(eigenvaluesV)[::-1]
    eigenvaluesV_sorted = eigenvaluesV[eigenvaluesV_sorted_ids]
    v = eigenvectorsV[:, eigenvaluesV_sorted_ids]
    # print("v.shape: ", v.shape)

    ### SIGMA
    s = np.diag(np.sqrt(eigenvaluesV_sorted))
    # print("s.shape: ", s.shape)
    s = add_padding(s, image_reshaped.shape[0], image_reshaped.shape[1])
    # print("s_pad.shape: ", s.shape)

    s_T = add_padding(s, image_reshaped.shape[0], image_reshaped.shape[1]).T
    s_plus = np.where(s_T == 0, 0, 1 / s_T)
    # print("s_plus.shape: ", s_plus.shape)

    ### U
    u = np.dot(image_reshaped, np.dot(v, s_plus))
    # print("u.shape: ", u.shape)

    compressed = np.dot(u[:, :k], np.dot(s[:k, :k], v.T[:k, :]))
    # compressed = np.interp(compressed, (compressed.min(), compressed.max()), (0, 1))
    compressed_reshaped = compressed.reshape(shape)

    full_variantion = np.diag(s).sum()
    selected_variantion = np.diag(s[:k]).sum()
    variantion_ratio = selected_variantion / full_variantion
    # print(f"Compression {100-variantion_ratio*100}%")

    fig, axes = plt.subplots(1, 2, figsize=(10, 7))
    ax = axes.ravel()
    ax[0].imshow(img)
    ax[0].set_title("Original")
    ax[1].imshow(compressed_reshaped)
    ax[1].set_title(
        f"Compression: {np.around(100-variantion_ratio*100)}% {k}/{s.shape[0]} eigenvalues"
    )

    fig.tight_layout()
    if args.output_file:
        plt.savefig(args.output_file)
    else:
        plt.show()


if __name__ == "__main__":
    img = img_as_float(io.imread(args.file))
    if args.k:
        k = args.k
    else:
        k = img.shape[0]

    if args.svd == "custom":
        custom_svd(img, k)
    elif args.svd == "library":
        library_svd(img, k)
