import numpy as np
import matplotlib.pyplot as plt

#---------------------------------------------------------------
# Convenient plotting functions for digit images such as
# those from sklearn.datasets.load_digits and
# sklearn.datasets.fetch_openml
#---------------------------------------------------------------

class DigitPlotter:
    def __init__(self, img_size_px):
        self.img_size_px = img_size_px
        self.is_binary_cm = True

    def plot_single(self, image, label):
        if self.is_binary_cm:
            plt.imshow(
                np.reshape(image, self.img_size_px),
                cmap=plt.cm.binary,
                interpolation="nearest"
            )
        else:
            plt.imshow(
                np.reshape(image, self.img_size_px),
                cmap=plt.cm.gray
            )
        plt.title('Label: {}\n'.format(label), fontsize = 20)

    def plot_multiple(self, images, labels):
        num_plots = len(images)
        num_cols = 5
        num_rows = int(num_plots / num_cols) + 1
        plt.figure(figsize=(20,5*num_rows))
        for index, (image, label) in enumerate(zip(images, labels)):
            plt.subplot(num_rows, num_cols, index + 1)
            self.plot_single(image, label)


#---------------------------------------------------------------
# Convenient plotting functions for SVM
#---------------------------------------------------------------
def plot_svc_decision_boundary(svm_clf, xmin, xmax):
    w = svm_clf.coef_[0]
    b = svm_clf.intercept_[0]

    # At the decision boundary, w0*x0 + w1*x1 + b = 0
    # => x1 = -w0/w1 * x0 - b/w1
    x0 = np.linspace(xmin, xmax, 200)
    decision_boundary = -w[0]/w[1] * x0 - b/w[1]

    margin = 1/w[1]
    gutter_up = decision_boundary + margin
    gutter_down = decision_boundary - margin

    svs = svm_clf.support_vectors_
    plt.scatter(svs[:, 0], svs[:, 1], s=180, facecolors='#FFAAAA')
    plt.plot(x0, decision_boundary, "k-", linewidth=2)
    plt.plot(x0, gutter_up, "k--", linewidth=2)
    plt.plot(x0, gutter_down, "k--", linewidth=2)
