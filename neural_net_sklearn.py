# # 8 bit
# # print('\u001B[48;5;99m  ', end="")
# # print('\u001B[0m')

# print('Color spectrum')
# print('\u001B[40m  '+'\u001B[100m  '+'\u001B[47m  '+'\u001B[107m  '+'\u001B[0m', end='\n\n')

from sklearn.neural_network import MLPClassifier
from sklearn import datasets

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

# learn_dataset(datasets.load_diabetes())
# learn_dataset(datasets.load_breast_cancer())

