import numpy as np
from scipy.linalg import solve
from matplotlib import pyplot as plt


def find_polynom(x, f, coefs):
	b = f(x)
	A = coefs(x)
	w = solve(A, b)
	
	return lambda x : sum(x ** i * w[i] for i in range(0, len(w)))


def plot(f, fnew, x):
	plt.plot(x, fnew(x), 'o', x, f(x), '-')
	plt.show()


def main():
	f = lambda x : np.sin(x / 5.) * np.exp(x / 10.) + 5 * np.exp(-x / 2.)
	coefs = lambda x : np.array([x ** i for i in range(0, len(x))], dtype=float).T
	fnew = find_polynom(np.array([1, 15]), f, coefs)
	plot(f, fnew, np.arange(1, 15, 0.1))
	fnew = find_polynom(np.array([1, 8, 15]), f, coefs)
	plot(f, fnew, np.arange(1, 15, 0.1))
	fnew = find_polynom(np.array([1, 4, 10, 15]), f, coefs)
	plot(f, fnew, np.arange(1, 15, 0.1))
	# for submission
	x = np.array([1, 4, 10, 15])
	np.savetxt('submission-2.txt', solve(coefs(x), f(x))[None], fmt='%.2f')


if __name__ == '__main__':
	main()