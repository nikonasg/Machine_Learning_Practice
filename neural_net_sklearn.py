# # 8 bit
# # print('\u001B[48;5;99m  ', end="")
# # print('\u001B[0m')

# print('Color spectrum')
# print('\u001B[40m  '+'\u001B[100m  '+'\u001B[47m  '+'\u001B[107m  '+'\u001B[0m', end='\n\n')

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

def example_digits():
  print('Example digits')
  for i in range(10):
    print(i)
    display_digit(i)

def learn_digits():
  print(digits_set.DESCR)
  classifier = MLPClassifier() #This is the neural net
  validation_size = 10
  classifier.fit(digits_set.data[validation_size:], digits_set.target[validation_size:])
  results = classifier.predict(digits_set.data[:validation_size])
  for i in range(len(results)):
    print('Neural Net guessed: ' + str(results[i]))
    print('Actual value: ' + str(digits_set.target[i]))
    display_digit(i)

def learn_dataset(dataset):
  print(dataset.DESCR)
  classifier = MLPClassifier() #This is the neural net
  validation_size = 37
  classifier.fit(dataset.data[validation_size:], dataset.target[validation_size:])
  results = classifier.predict(dataset.data[:validation_size])
  num_correct = 0
  for i in range(len(results)):
    print('Trial #' + str(i))
    print('Neural Net guessed: ' + str(results[i]))
    print('Actual value: ' + str(dataset.target[i]))
    num_correct += 1 if results[i] == dataset.target[i] else 0
    print()
  percent_correct = '%.2f' % (num_correct / validation_size * 100)
  print(end=('Percentage correct: ' + percent_correct + '%\n') if percent_correct != '0.00' else "")

# example_digits()
learn_digits()
# learn_dataset(datasets.load_diabetes())
# learn_dataset(datasets.load_breast_cancer())

