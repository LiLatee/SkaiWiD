#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def vectors_uniform(k):
    """Uniformly generates k vectors."""
    vectors = []
    for a in np.linspace(0, 2 * np.pi, k, endpoint=False):
        vectors.append(2 * np.array([np.sin(a), np.cos(a)]))
    return vectors


def visualize_transformation(A, vectors):
    """Plots original and transformed vectors for a given 2x2 transformation matrix A and a list of 2D vectors."""
    for i, v in enumerate(vectors):
        # Plot original vector.
        plt.quiver(0.0, 0.0, v[0], v[1], width=0.008, color="blue", scale_units='xy', angles='xy', scale=1,
                   zorder=4)
        plt.text(v[0]/2 + 0.25, v[1]/2, "v{0}".format(i), color="blue")

        # Plot transformed vector.
        tv = A.dot(v)
        plt.quiver(0.0, 0.0, tv[0], tv[1], width=0.005, color="magenta", scale_units='xy', angles='xy', scale=1,
                   zorder=4)
        plt.text(tv[0] / 2 + 0.25, tv[1] / 2, "v{0}'".format(i), color="magenta")
    plt.xlim([-6, 6])
    plt.ylim([-6, 6])
    plt.margins(0.05)
    # Plot eigenvectors
    plot_eigenvectors(A)
    plt.show()


def visualize_vectors(vectors, color="green"):
    """Plots all vectors in the list."""
    for i, v in enumerate(vectors):
        plt.quiver(0.0, 0.0, v[0], v[1], width=0.006, color=color, scale_units='xy', angles='xy', scale=1,
                   zorder=4)
        plt.text(v[0] / 2 + 0.25, v[1] / 2, "eigv{0}".format(i), color=color)


def plot_eigenvectors(A):
    """Plots all eigenvectors of the given 2x2 matrix A."""
    # TODO: Zad. 4.1. Oblicz wektory własne A. Możesz wykorzystać funkcję np.linalg.eig
    values, eigvec = np.linalg.eig(A)
#     print(values)
#     print(eigvec)
#     print(np.transpose(eigvec))
#     eigvec = []
    # TODO: Zad. 4.1. Upewnij się poprzez analizę wykresów, że rysowane są poprawne wektory własne (łatwo tu o pomyłkę).
#     visualize_vectors(eigvec)
    visualize_vectors(np.transpose(eigvec))


def EVD_decomposition(A):
    # TODO: Zad. 4.2. Uzupełnij funkcję tak by obliczała rozkład EVD zgodnie z zadaniem.
    values, eigvec = np.linalg.eig(A)
    K = eigvec
    print("A: \n", A)
    print("K: \n", K)
    L = np.diag(values)
    print("L: \n", L)
    inv_K =  np.linalg.inv(K)
    print("K^-1: \n", inv_K)
    A2 = np.matmul(np.matmul(K,L), inv_K)
    print("A2: \n", A2)
    print("="*100)
    pass


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm
   
def plot_attractors(A, vectors):
    # TODO: Zad. 4.3. Uzupełnij funkcję tak by generowała wykres z atraktorami.
    import random 
    import matplotlib.colors as mcolors
    values, eigvecs = np.linalg.eig(A)
    eigvecs = np.transpose(eigvecs)
    
    # print("eigenvalues: \n", values)
    # print("eigvecs: \n", eigvecs)
    eigvecs = [np.array(el) for el in eigvecs]
    # Dodajemy wektory przeciwne do wektorów własnych.
    temp_eigvecs = eigvecs.copy()
    for el in temp_eigvecs:
        eigvecs.append(el*-1)
        
    plt.figure(figsize=(8,8))
    
    # Tworzymy kolory.
    colors = list(mcolors.TABLEAU_COLORS.values())
    if len(eigvecs) > len(colors):    
        colors = []
        for el in eigvecs:
            r = random.random()
            b = random.random()
            g = random.random()
            color = (r, g, b)
            colors.append(color)
    
    # Rysujemy wektory własne + wektory do nich przeciwne - krótsze wektory z ostrymi strzałkami.
    for i, el in enumerate(eigvecs):
        plt.quiver(0.0, 0.0, el[0], el[1], width=0.006, color=colors[i], scale_units='xy', angles='xy', scale=1, zorder=4,
           headwidth=10, headlength=10)
        if i < len(values):
            plt.text(el[0], el[1], round(values[i], 2), color="black", zorder=5)
            
   # Rysujemy podane w argumencie funkcji wektory w kolorze wektora własnego (lub przeciwnego), jeśli jest to ich atraktor.
    for v in vectors:
        v = normalize(v)
        temp = v.copy()
        make_black = True
        for i, eigvec in enumerate(eigvecs): 
            for j in range(100):
                temp = normalize(A.dot(temp))
                if (np.allclose(temp, eigvec, atol=0.01)):
                    plt.quiver(0.0, 0.0, v[0], v[1], width=0.006, color=colors[i], scale_units='xy', angles='xy', scale=1, zorder=4)
                    make_black = False
                    break

            if make_black:
                plt.quiver(0.0, 0.0, v[0], v[1], width=0.006, color='black', scale_units='xy', angles='xy', scale=1, zorder=4)


    plt.xlim([-3, 3])
    plt.ylim([-3, 3])
    plt.margins(0.05)
    plt.grid()
    plt.show()


def show_eigen_info(A, vectors):
    EVD_decomposition(A)
    visualize_transformation(A, vectors)
    plot_attractors(A, vectors)


if __name__ == "__main__":
    vectors = vectors_uniform(k=16)

    A = np.array([[2, 0],
                  [0, 2]])
    show_eigen_info(A, vectors)


    A = np.array([[-1, 2],
                  [2, 1]])
    show_eigen_info(A, vectors)


    A = np.array([[3, 1],
                  [0, 2]])
    show_eigen_info(A, vectors)


    A = np.array([[2, -1],
                  [1, 4]])
    show_eigen_info(A, vectors)