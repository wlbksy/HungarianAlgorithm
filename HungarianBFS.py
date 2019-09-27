import numpy as np
from abc import ABC, abstractmethod


class HungarianBFS(ABC):
    def __init__(self, matrix):
        height, width = matrix.shape

        self.matrix = matrix

        self.thin = height > width

        if self.thin:
            self.height = width
            self.width = height
        else:
            self.height = height
            self.width = width

    def solve(self):
        self.match = [-1 for right in range(self.width)]

        potential_left = [0 for left in range(self.height)]
        potential_right = [0 for right in range(self.width)]
        for left in range(self.height):
            prev = [-1 for right in range(self.width)]
            visited = [False for right in range(self.width)]
            slack = [self.get_bound_value() for right in range(self.width)]
            marked_left = left
            marked_right = -1
            j = 0
            while marked_left != -1:
                j = -1
                for right in range(self.width):
                    if visited[right]:
                        continue
                    diff = self.get_matrix_element(marked_left, right) - \
                        potential_left[marked_left] - potential_right[right]
                    if self.compare(diff, slack[right]):
                        slack[right] = diff
                        prev[right] = marked_right
                    if j == -1 or self.compare(slack[right], slack[j]):
                        j = right

                delta = slack[j]
                for right in range(self.width):
                    if visited[right]:
                        potential_left[self.match[right]] += delta
                        potential_right[right] -= delta
                    else:
                        slack[right] -= delta
                potential_left[left] += delta

                visited[j] = True
                marked_right = j
                marked_left = self.match[j]
            while prev[j] != -1:
                self.match[j] = self.match[prev[j]]
                j = prev[j]
            self.match[j] = left

    @abstractmethod
    def get_bound_value(self):
        raise NotImplementedError

    @abstractmethod
    def compare(self, a, b):
        raise NotImplementedError

    def get_matches(self):
        if self.thin:
            return [(right, self.match[right]) for right in range(self.width) if self.match[right] != -1]
        return [(self.match[right], right) for right in range(self.width) if self.match[right] != -1]

    def get_matrix_element(self, row, col):
        if self.thin:
            return self.matrix[col, row]
        return self.matrix[row, col]


class MinCostHungarian(HungarianBFS):
    def get_bound_value(self):
        return np.inf

    def compare(self, a, b):
        return a < b


class MaxCostHungarian(HungarianBFS):
    def get_bound_value(self):
        return -np.inf

    def compare(self, a, b):
        return a > b
