import numpy as np


data = np.load('C:/Users/skyla/Desktop/FocuSSU_Project123123/my_face.npy')


np.savetxt('C:/Users/skyla/Desktop/FocuSSU_Project123123/my_face.csv', data, delimiter=',')
