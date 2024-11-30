import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_rede_cristalina(vetores, nome_rede):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
 
    ax.scatter(0, 0, 0, c='r', label='Átomo base (0, 0, 0)')
    
    pontos = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                ponto = i * vetores[0] + j * vetores[1] + k * vetores[2]
                pontos.append(ponto)
                ax.scatter(ponto[0], ponto[1], ponto[2], c='b')
    

    for p in pontos:
        for q in pontos:
            if np.linalg.norm(p - q) <= np.linalg.norm(vetores[0]):
                ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], c='k', alpha=0.5)
    
    ax.set_title(f"Estrutura: {nome_rede}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.legend()
    plt.show()

def cubica_simples():
    a = 1
    return np.array([a, 0, 0]), np.array([0, a, 0]), np.array([0, 0, a])

def tetragonal_p():
    a, c = 1, 2
    return np.array([a, 0, 0]), np.array([0, a, 0]), np.array([0, 0, c])

def ortorrombica_p():
    a, b, c = 1, 2, 3
    return np.array([a, 0, 0]), np.array([0, b, 0]), np.array([0, 0, c])

def monoclinica_p():
    a, b, c, beta = 1, 2, 3, np.radians(110)
    return np.array([a, 0, 0]), np.array([0, b, 0]), np.array([c * np.cos(beta), 0, c * np.sin(beta)])

def triclina():
    a, b, c, alpha, beta, gamma = 1, 2, 3, np.radians(70), np.radians(80), np.radians(85)
    ax = a
    bx = b * np.cos(gamma)
    by = b * np.sin(gamma)
    cx = c * np.cos(beta)
    cy = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
    cz = np.sqrt(c**2 - cx**2 - cy**2)
    return np.array([ax, 0, 0]), np.array([bx, by, 0]), np.array([cx, cy, cz])

def trigonal_r():
    a = 1
    alpha = np.radians(60)
    return (
        np.array([a, 0, 0]),
        np.array([a * np.cos(alpha), a * np.sin(alpha), 0]),
        np.array([0, 0, a])
    )

def hexagonal_p():
    a, c = 1, 2
    return (
        np.array([a, 0, 0]),
        np.array([-a / 2, a * np.sqrt(3) / 2, 0]),
        np.array([0, 0, c])
    )

estruturas = {
    "Cúbica Simples": cubica_simples,
    "Tetragonal P": tetragonal_p,
    "Ortorrômbica P": ortorrombica_p,
    "Monoclínica P": monoclinica_p,
    "Triclínica": triclina,
    "Trigonal R": trigonal_r,
    "Hexagonal P": hexagonal_p
}

for nome, func in estruturas.items():
    vetores = func()
    plot_rede_cristalina(vetores, nome)
