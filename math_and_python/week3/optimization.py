import math
import numpy as np
from scipy.optimize import minimize, differential_evolution
from matplotlib import pyplot as plt


def optimization_smooth(f, p1, p2):
	smooth_opt1 = minimize(fun=f, x0=[p1], method='BFGS')
	smooth_opt2 = minimize(fun=f, x0=[p2], method='BFGS')
	print("\n{}\n\n{}".format(smooth_opt1, smooth_opt2))
	np.savetxt('submission-1.txt', np.append(smooth_opt1.fun, smooth_opt2.fun)[None], fmt='%.2f')


def optimization_global(f, bounds):
	diff_opt = differential_evolution(f, [bounds])
	print("\n{}".format(diff_opt))
	np.savetxt('submission-2.txt', diff_opt.fun, fmt='%.2f')


def optimization_nonsmooth(f, p, bounds):
	bfgs_opt = minimize(fun=f, x0=p, method='BFGS')
	diff_opt = differential_evolution(f, [bounds])
	print("\n{}\n\n{}".format(bfgs_opt, diff_opt))
	np.savetxt('submission-3.txt', np.append(bfgs_opt.fun, diff_opt.fun)[None], fmt='%.2f')


def main():
	f = np.vectorize(lambda x : math.sin(x / 5.) * math.exp(x / 10.) + 5. * math.exp(-x / 2.))
	h = np.vectorize(lambda x : int(f(x)))
	x = np.arange(1, 30, 0.1)
	optimization_smooth(f, 2, 30)
	optimization_global(f, (1, 30))
	optimization_nonsmooth(h, 30, (1, 30))
	plt.plot(x, f(x), 'o', x, h(x), '-')
	plt.show()

if __name__ == '__main__':
	main()