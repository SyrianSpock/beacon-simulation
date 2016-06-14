import numpy as np


class EKF(object):
    def __init__(self, n, f, F, h, H):
        self.x = np.zeros([n, 1])
        self.P = np.eye(n)
        self.f = f
        self.F = F
        self.h = h
        self.H = H

    def predict(self, u, Q, dt):
        self.x = self.f(self.x, u, dt)
        F = self.F(self.x, u, dt)
        self.P = np.dot(F, np.dot(self.P, F.transpose())) + Q
        self.inspect_F = F

    def measure(self, z, R):
        y = z - self.h(self.x)
        H = self.H(self.x)
        S = np.dot(H, np.dot(self.P, H.transpose())) + R
        K = np.dot(np.dot(self.P, H.transpose()), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        n = len(self.P)
        self.P = np.dot((np.eye(n) - np.dot(K, H)), self.P)
        self.inspect_H = H
        self.inspect_K = K

    def reset(self, x, P):
        self.x = x
        self.P = P
