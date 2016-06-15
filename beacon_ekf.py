import numpy as np
from ekf import EKF

PROCESS_NOISE_VARIANCE = 1e-4

class BeaconEKF(EKF):
    def __init__(self, b1, b2, b3, dt):
        EKF.__init__(self, 6, self.f, self.F, self.h, self.H)
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3
        self.dt = dt

        # self.Q = self.get_Q(dt)
        self.Q = PROCESS_NOISE_VARIANCE * np.eye(6)

    def predict(self, u):
        EKF.predict(self, u, self.Q, self.dt)

    def get_Q(self, dt):
        F_cont = np.array([[0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 1],
                           [0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0]])
        G = np.array([[0, 0, 0],
                      [0, 0, 0],
                      [0, 0, 0],
                      [1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])
        W = PROCESS_NOISE_VARIANCE * np.eye(3)

        A = dt * np.concatenate([np.concatenate([-F_cont, np.dot(G, np.dot(W, G.T))], axis=1),
                                 np.concatenate([np.zeros_like(F_cont), F_cont], axis=1)], axis=0)
        B = np.eye(len(A)) + A + 0.5 * np.dot(A, A) # approx expm(A)

        Q = np.dot(B[self.n:,self.n:].T, B[:self.n,self.n:])

        return Q

    def f(self, x_k, u_k, dt):
        x_k_1 = x_k

        x_k_1[0] = x_k[0] + dt * x_k[3]
        x_k_1[1] = x_k[1] + dt * x_k[4]
        x_k_1[2] = x_k[2] + dt * x_k[5]

        return x_k_1

    def F(self, x_k, u_k, dt):
        F = np.diag([1,1,1,1,1,1])

        F[0][3] = dt
        F[1][4] = dt
        F[2][5] = dt

        return F

    def h(self, x):
        z = np.zeros([3, 1])

        z[0] = np.linalg.norm(x[0:3].T - self.b1)
        z[1] = np.linalg.norm(x[0:3].T - self.b2)
        z[2] = np.linalg.norm(x[0:3].T - self.b3)

        return z

    def H(self, x):
        H = np.zeros([3, self.n])

        H[0][0] = (x[0] - self.b1[0]) / np.linalg.norm(self.b1 - x[0:3])
        H[0][1] = (x[1] - self.b1[1]) / np.linalg.norm(self.b1 - x[0:3])
        H[0][2] = (x[2] - self.b1[2]) / np.linalg.norm(self.b1 - x[0:3])

        H[1][0] = (x[0] - self.b2[0]) / np.linalg.norm(self.b2 - x[0:3])
        H[1][1] = (x[1] - self.b2[1]) / np.linalg.norm(self.b2 - x[0:3])
        H[1][2] = (x[2] - self.b2[2]) / np.linalg.norm(self.b2 - x[0:3])

        H[2][0] = (x[0] - self.b3[0]) / np.linalg.norm(self.b3 - x[0:3])
        H[2][1] = (x[1] - self.b3[1]) / np.linalg.norm(self.b3 - x[0:3])
        H[2][2] = (x[2] - self.b3[2]) / np.linalg.norm(self.b3 - x[0:3])

        return H
