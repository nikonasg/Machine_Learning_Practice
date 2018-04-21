from sklearn.neural_network import MLPClassifier
from sklearn import datasets

digits_set = datasets.load_digits()

def display_digit(digit_index):
	dark_black = '\u001B[40m  '
	light_black = '\u001B[100m  '
	dark_white = '\u001B[47m  '
	light_white = '\u001B[107m  '
	reset_color = '\u001B[0m'
	for r in range(8):
		for c in range(8):
			i = r * 8 + c
			print(dark_black if digits_set.data[digit_index][i] <= 4.0 else light_black if digits_set.data[digit_index][i] <= 8.0 else dark_white if digits_set.data[digit_index][i] <= 12.0 else light_white, end="")
		print(reset_color)
	print()

def learn_digits():
	# print(digits_set.DESCR)
	classifier = MLPClassifier() #This is the neural net
	test_size = 10
	classifier.fit(digits_set.data[test_size:], digits_set.target[test_size:])
	results = classifier.predict(digits_set.data[:test_size])
	for i in range(len(results)):
		print('Neural Net guessed: ' + str(results[i]))
		print('Actual value: ' + str(digits_set.target[i]))
		display_digit(i)

learn_digits()
