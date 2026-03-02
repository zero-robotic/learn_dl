import numpy as np
from d2l import plot

x = np.arange(0, 3, 0.1)
plot(x, [x ** 3 - 1 / x, 3 * x * x + x ** -2], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])
