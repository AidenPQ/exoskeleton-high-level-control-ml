import numpy as np
from scipy.interpolate import make_interp_spline


class CurveReconstruction:

    def recompose_keypoints(self, extremum_pos, extremum_values, curvature_inversion_pos, curvature_inversion_values, begin_point, begin_point_value, more_points_pos, more_points_values, time_norm):
        pos_list = [begin_point]
        pos_list.extend(extremum_pos)
        pos_list.extend(curvature_inversion_pos)
        pos_list.extend(more_points_pos)
        sorted_pos_list = sorted(pos_list)
        sorted_values_list = []

        for i in range(len(sorted_pos_list)):
            ind = pos_list.index(sorted_pos_list[i])
            if ind == 0:
                sorted_values_list.append(begin_point_value)
            elif (ind >= 1) & (ind <= len(extremum_pos)):
                sorted_values_list.append(extremum_values[ind - 1])
            elif len(extremum_pos) < ind <= len(extremum_pos) + len(curvature_inversion_pos):
                real_ind = ind - (1 + len(extremum_pos))
                sorted_values_list.append(curvature_inversion_values[real_ind])
            elif ind > (len(extremum_pos) + len(curvature_inversion_pos)):
                real_ind = ind - (1 + len(extremum_pos) + len(curvature_inversion_pos))
                sorted_values_list.append(more_points_values[real_ind])

        sorted_pos_list.append(time_norm[-1])
        sorted_values_list.append(begin_point_value)
        return sorted_pos_list, sorted_values_list

    def interpolation_sorted_values(self, sorted_pos, sorted_values, interp_degree):
        spline_interpolation = make_interp_spline(sorted_pos, sorted_values, k=interp_degree)

        return spline_interpolation

    def interpolation(self, extremum_pos, extremum_values, curvature_inversion_pos, curvature_inversion_values, begin_point, begin_point_value, more_points_pos, more_points_values,
                      interp_degree, time_norm):
        sorted_pos_list, sorted_values_list = self.recompose_keypoints(extremum_pos, extremum_values, curvature_inversion_pos, curvature_inversion_values, begin_point, begin_point_value, more_points_pos, more_points_values, time_norm)

        spline_interpolation = make_interp_spline(sorted_pos_list, sorted_values_list, k=interp_degree)

        # Generate points for plotting the interpolated curve
        x_interp = np.linspace(0, 100, len(time_norm))

        y_interp = spline_interpolation(x_interp)
        return x_interp, y_interp


