import argparse
import numpy as np
from skimage import io
from skimage import img_as_float, img_as_uint
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
# parser.add_argument("-v", "--verbose", help="increase output verbosity")
parser.add_argument("-f", "--file", help="plik z oryginalnym obrazem")
parser.add_argument("-out", "--output_file", help="nazwa pliku wyjściowego")
parser.add_argument(
    "-svd", help="implementacja SVD do użycia", choices=["custom", "library"]
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

    print("u.shape: ", u.shape)
    print("s.shape: ", s.shape)
    print("v.shape: ", v.shape)

    full_variantion = np.diag(s).sum()
    selected_variantion = np.diag(s[:k]).sum()
    variantion_ratio = selected_variantion / full_variantion
    print(f"Compression {100-variantion_ratio*100}%")

    fig, axes = plt.subplots(1, 2, figsize=(10, 7))
    ax = axes.ravel()
    ax[0].imshow(img)
    ax[1].imshow(compressed_reshaped)
    fig.tight_layout()
    plt.show()


# def custom_svd(img, k):
#     shape = img.shape
#     print("shape: ", shape)
#     image_reshaped = img.reshape((shape[0], shape[1] * 3))
#     print("reshaped: ", image_reshaped.shape)

#     k = min(image_reshaped.shape[0], image_reshaped.shape[1])
#     ### U
#     img_imgT = np.dot(image_reshaped, image_reshaped.T)
#     eigenvaluesU, eigenvectorsU = np.linalg.eigh(img_imgT)
#     eigenvaluesU_sorted_ids = np.argsort(eigenvaluesU)[::-1]
#     eigenvaluesU_sorted = eigenvaluesU[eigenvaluesU_sorted_ids]
#     u = eigenvectorsU[:, eigenvaluesU_sorted_ids][: image_reshaped.shape[0], :k]
#     u = u * -1
#     ### SIGMA
#     eigenvaluesU_sorted = [x for x in eigenvaluesU_sorted if x > 0]
#     s = np.diag(np.sqrt(eigenvaluesU_sorted))

#     ### V
#     imgT_img = np.dot(image_reshaped.T, image_reshaped)
#     eigenvaluesV, eigenvectorsV = np.linalg.eigh(imgT_img)
#     eigenvaluesV_sorted_ids = np.argsort(eigenvaluesV)[::-1]
#     eigenvaluesV_sorted = eigenvaluesV[eigenvaluesV_sorted_ids]
#     v = eigenvectorsV[:, eigenvaluesV_sorted_ids][:k, : image_reshaped.shape[1]]
#     # v = abs(v)
#     # s = np.diag(np.sqrt(eigenvaluesV_sorted))[: image_reshaped.shape[0], :]

#     U, S, V = np.linalg.svd(image_reshaped, full_matrices=False)

#     print("LIBRARY: ")
#     print("u.shape: ", U.shape)
#     print("s.shape: ", S.shape)
#     print("v.shape: ", V.shape)

#     # u = u[:, :k]
#     # s = s[:, :k]
#     # v = v.T[:k, :]
#     print("CUSTOM: ")
#     print("u.shape: ", u.shape)
#     print("s.shape: ", s.shape)
#     print("v.shape: ", v.shape)

#     compressed = np.dot(U, np.dot(s, V))
#     print(compressed.min())
#     print(compressed.max())
#     # compressed = abs(compressed)
#     # compressed = img_as_uint(compressed)
#     # compressed = np.dot(U, np.dot(np.diag(S), V))
#     compressed_reshaped = compressed.reshape(shape)

#     fig, axes = plt.subplots(1, 2, figsize=(10, 7))
#     ax = axes.ravel()
#     ax[0].imshow(img)
#     ax[1].imshow(compressed_reshaped)
#     fig.tight_layout()
#     plt.show()


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
    print("shape: ", shape)
    image_reshaped = img.reshape((shape[0], shape[1] * 3))
    print("reshaped: ", image_reshaped.shape)

    k = min(image_reshaped.shape[0], image_reshaped.shape[1])

    ### V
    imgT_img = np.dot(image_reshaped.T, image_reshaped)
    eigenvaluesV, eigenvectorsV = np.linalg.eigh(imgT_img)
    eigenvaluesV_sorted_ids = np.argsort(eigenvaluesV)[::-1]
    eigenvaluesV_sorted = eigenvaluesV[eigenvaluesV_sorted_ids]
    v = eigenvectorsV[:, eigenvaluesV_sorted_ids]
    print("v.shape: ", v.shape)

    s = np.diag(eigenvaluesV_sorted)
    s = add_padding(s, image_reshaped.shape[0], image_reshaped.shape[1]).T
    print("s.shape: ", s.shape)

    ### SIGMA
    s_plus = np.where(s == 0, 0, 1 / s)
    print("s_plus.shape: ", s_plus.shape)

    ### U
    u = np.dot(image_reshaped, np.dot(v, s_plus))
    print("v.shape: ", v.shape)

    compressed = np.dot(u, np.dot(s, v))
    print(compressed.min())
    print(compressed.max())
    compressed_reshaped = compressed.reshape(shape)

    fig, axes = plt.subplots(1, 2, figsize=(10, 7))
    ax = axes.ravel()
    ax[0].imshow(img)
    ax[1].imshow(compressed_reshaped)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    img = img_as_float(io.imread(args.file))
    k = args.k

    # library_svd(img, k)
    custom_svd(img, k)

    # img_imgT = np.dot(arr, arr.T)
    # eigenvaluesU, eigenvectorsU = np.linalg.eigh(img_imgT)
    # eigenvaluesU_sorted_ids = np.argsort(eigenvaluesU)[::-1]
    # eigenvaluesU_sorted = eigenvaluesU[eigenvaluesU_sorted_ids]
    # u = eigenvectorsU[:, eigenvaluesU_sorted_ids]

    # s = np.diag(np.sqrt(eigenvaluesU_sorted))[:, : arr.shape[1]]

    # imgT_img = np.dot(arr.T, arr)
    # eigenvaluesV, eigenvectorsV = np.linalg.eigh(imgT_img)
    # eigenvaluesV_sorted_ids = np.argsort(eigenvaluesV)[::-1]
    # eigenvaluesV_sorted = eigenvaluesV[eigenvaluesV_sorted_ids]
    # v = eigenvectorsV[:, eigenvaluesV_sorted_ids]

    # print("u.shape: ", u.shape)
    # print(u)
    # print("s.shape: ", s.shape)
    # print(s)
    # print("vT.shape: ", v.T.shape)
    # print(v)

    # compressed = np.dot(u, np.dot(s, v.T))
    # print(compressed)
