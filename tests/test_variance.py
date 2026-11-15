"""Narrow off-origin slices must not lose variance to raw-moment cancellation."""
from decimal import Decimal as D, localcontext
import math
import unittest
from flowlab.conditional import conditional_slice, window_audit


class Uniform:
    def log_prob(self, point):
        return 0.


def decimal_variance(report):
    # Independent high-precision monomial integration in absolute coordinates.
    with localcontext() as ctx:
        ctx.prec = 90
        xs = list(map(D.from_float, report['grid']))
        logs = report['joint_log_density']
        heights = [D.from_float(math.exp(v-max(logs))) for v in logs]
        moments = []
        for power in range(3):
            total = D(0)
            for i in range(len(xs)-1):
                lo, hi = xs[i:i+2]
                slope = (heights[i+1]-heights[i])/(hi-lo)
                intercept = heights[i]-slope*lo
                total += intercept*(hi**(power+1)-lo**(power+1))/D(power+1)
                total += slope*(hi**(power+2)-lo**(power+2))/D(power+2)
            moments.append(total)
        return float(moments[2]/moments[0]-(moments[1]/moments[0])**2)


class Variance(unittest.TestCase):
    def test_uniform_narrow_both_signs_and_grid_sizes(self):
        for low in [-99., 0., 99.]:
            for width in [2e-6, 1e-4, .01, .5]:
                for size in [2, 17, 513]:
                    high = low+width
                    report = conditional_slice(Uniform(), 0, 0., low, high, size)
                    expected = (high-low)**2/12
                    self.assertLess(abs(report['variance']/expected-1), 2e-12)
                    self.assertGreater(report['variance'], 0.)

    def test_linear_segments_against_decimal_raw_moments(self):
        count = 0
        for low in [-99., 0., 99.]:
            for width in [2e-6, 1e-4, .1]:
                for sign in [-1, 1]:
                    for size in [2, 17, 129]:
                        class Sloped:
                            def log_prob(self, point):
                                return math.log(2+sign*(point[1]-low)/width)
                        report = conditional_slice(Sloped(), 0, 0., low, low+width, size)
                        expected = decimal_variance(report)
                        self.assertLess(abs(report['variance']/expected-1), 2e-11)
                        count += 1
        self.assertEqual(count, 54)

    def test_aligned_windows_keep_their_own_nonzero_variances(self):
        for axis in [0, 1]:
            report = window_audit(Uniform(), axis, 0., 99., 99.000002, 33, 8, 24)
            for key in ['outer', 'inner']:
                distribution = report[key]
                lo, hi = distribution['interval']
                expected = (hi-lo)**2/12
                self.assertLess(abs(distribution['variance']/expected-1), 2e-12)
            self.assertEqual(report['model_evaluations'], 33)


if __name__ == '__main__':
    unittest.main()
