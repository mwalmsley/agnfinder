# import pytest

# import numpy as np
# import pandas as pd

# from agnfinder import forward_model


# @pytest.fixture
# def mag_col():
#     return 'i_sdss'


# @pytest.fixture
# def templates(mag_col):
#     return pd.DataFrame(data=[
#         {'EB_V': 0.1, 'z': 0.01, mag_col: 12.},
#         {'EB_V': 0.1, 'z': 0.02, mag_col: 13.},
#         {'EB_V': 0.2, 'z': 0.01, mag_col: 14.},
#         {'EB_V': 0.2, 'z': 0.02, mag_col: 15.},
#     ])


# @pytest.fixture
# def grid():
#     return np.array([[0., 1., 2.],[0., 2., 4.]])

# @pytest.fixture
# def axes():
#     return ([1., 2.], [1., 2., 3.])

# def test_construct_grid(templates, mag_col):
#     axes, grid = forward_model.construct_grid(templates, mag_col, ['EB_V', 'z'])
#     assert grid.shape == (2, 2)
#     assert grid[0, 0] == 12
#     assert grid[1, 1] == 15
#     assert axes == [[0.1, 0.2], [0.01, 0.02]]

# def test_interpolate_templates(monkeypatch, templates, mag_col, axes, grid):
#     def mock_construct_grid(x, y):
#         return axes, grid
#     monkeypatch.setattr(forward_model, 'construct_grid', mock_construct_grid)
#     interp = forward_model.interpolate_templates(templates, mag_col)
#     assert interp((1.5, 2.)) == 1
