import re
import numpy as np
from scipy.spatial.distance import cosine


def main():
	text = []
	sentence = []
	words = {}
	d = 0
	n = 0
	file = open('sentences.txt', 'r')
	# out = open('submission-1.txt')

	for line in file.readlines():
		sentence = re.split('[^A-Za-z]', line.lower())
		sentence = list(filter(None, sentence))
		text.append(sentence)
		for word in sentence:
			if word in words:
				continue
			words[word] = d
			d += 1
		n += 1

	matrix = np.zeros((n, d))

	for i in range(0, len(text)):
		sentence = text[i]
		for word in sentence:
			matrix[i, words[word]] += 1

	distances = np.apply_along_axis(cosine, 1, matrix[1:, :], matrix[0])
	#submission
	np.savetxt('submission-1.txt', (distances.argsort()[:2] + 1)[None], fmt='%d')


if __name__ == '__main__':
	main()