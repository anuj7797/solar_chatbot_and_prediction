import matplotlib.pyplot as plt;

plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

class barchart:
    def bar(self):
        print("in bar");
        objects = ('SVM', 'ANN', 'LINEAR ', 'LOGISTIC')
        y_pos = np.arange(len(objects))
        performance = [10, 8, 6, 4]

        plt.bar(y_pos, performance, align='center', alpha=1)
        plt.xticks(y_pos, objects)
        plt.ylabel('Error')
        plt.title('Relative Error')

        plt.show()
