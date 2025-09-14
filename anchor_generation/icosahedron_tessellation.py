"""Uses icosahedron tessellation to generate a sphere mesh.

Each vertex of the icosahedron is used as an anchor point.
"""
import math

import numpy as np
import torch

from upright_anchor.anchor_generation.base import IAnchorGenerationService


class IcosahedronTessellation(IAnchorGenerationService):
    """Uses icosahedron tessellation to generate a sphere mesh.

    Each vertex of the icosahedron is used as an anchor point.
    """

    def __init__(self, n_anchors: int) -> None:
        """Initialize a new instance of IcosahedronTessellation.

        Args:
            n_anchors (int): The number of anchor points to generate.
        """
        self.n_anchors = n_anchors
        self.__middle_point_cache = {}
        # Generate the anchor points
        phi = (1 + np.sqrt(5)) / 2

        self.verts = [
            self.__normalize(-1, phi, 0),
            self.__normalize(1, phi, 0),
            self.__normalize(-1, -phi, 0),
            self.__normalize(1, -phi, 0),
            self.__normalize(0, -1, phi),
            self.__normalize(0, 1, phi),
            self.__normalize(0, -1, -phi),
            self.__normalize(0, 1, -phi),
            self.__normalize(phi, 0, -1),
            self.__normalize(phi, 0, 1),
            self.__normalize(-phi, 0, -1),
            self.__normalize(-phi, 0, 1),
        ]

        self.__faces = [
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ]

    @staticmethod
    def __normalize(x: float, y: float, z: float) -> tuple[float, float, float]:
        length = np.sqrt(x**2 + y**2 + z**2)
        return tuple(i / length for i in [x, y, z])

    def __middle_point(self, point_1: int, point_2: int) -> int:
        smaller_index = min(point_1, point_2)
        greater_index = max(point_1, point_2)

        key = f"{smaller_index}-{greater_index}"

        if key in self.__middle_point_cache:
            return self.__middle_point_cache[key]

        vert_1 = self.verts[point_1]
        vert_2 = self.verts[point_2]
        middle = [sum(i) / 2 for i in zip(vert_1, vert_2, strict=False)]

        self.verts.append(self.__normalize(*middle))

        index = len(self.verts) - 1
        self.__middle_point_cache[key] = index

        return index

    def generate(self) -> torch.Tensor:
        """Generate the anchor points using icosahedron tessellation."""
        # Calculate the number of subdivisions needed
        subdivisions = self.calculate_number_of_subdivisions()

        for _ in range(subdivisions):
            faces_subdiv = []

            for tri in self.__faces:
                v1 = self.__middle_point(tri[0], tri[1])
                v2 = self.__middle_point(tri[1], tri[2])
                v3 = self.__middle_point(tri[2], tri[0])

                faces_subdiv.extend([[tri[0], v1, v3], [tri[1], v2, v1], [tri[2], v3, v2], [v1, v2, v3]])

            self.__faces = faces_subdiv

        return torch.Tensor(self.verts)

    def calculate_number_of_subdivisions(self) -> int:
        """Calculate the number of subdivisions needed to generate the desired number of anchor points.

        The number of anchor points is given by the formula:
        N = 10 * 4^subdivisions + 2

        Returns:
            int: The number of subdivisions needed.

        Raises:
            ValueError: If the number of anchor points is invalid.
        """
        # Make sure the number of anchor points is valid
        if (self.n_anchors - 2) % 10 != 0:
            error_message = "Invalid number of anchor points."
            raise ValueError(error_message)

        # Calculate the number of subdivisions
        return int(math.log((self.n_anchors - 2) // 10, 4))
