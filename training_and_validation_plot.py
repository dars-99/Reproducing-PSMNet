import matplotlib.pyplot as plt
import torch
import pickle
import numpy as np



with open('loss_list', 'rb') as f:
    loss_list = pickle.load(f)


test_loss = []
training_loss = []
for i in range(len(loss_list)):
    training_loss.append(loss_list[i][1])
    test_loss.append(loss_list[i][2])
    
plt.plot(training_loss, label = 'Training loss')
plt.plot(test_loss, label = 'Validation loss (3px %)')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()
print(np.min(test_loss))
print(np.argmin(test_loss))
