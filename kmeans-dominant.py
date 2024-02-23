from tkinter import *
from tkinter import ttk
from tkinter import filedialog
import matplotlib.pyplot as plt
from PIL import ImageTk, Image
import cv2
import numpy as np


class Root(Tk):

    def __init__(self):
        super(Root, self).__init__()
        self.title("Tkinter Dialog Widget")
        self.minsize(300, 500)

        self.labelFrame = ttk.LabelFrame(self, text='Input Image')
        self.labelFrame.grid(column=0, row=1, padx=5, pady=5)

        self.labelFrame1 = ttk.LabelFrame(self, text='Path')
        self.labelFrame1.grid(column=0, row=3, padx=5, pady=5)

        self.labelFrame2 = ttk.LabelFrame(self, text='Cluster')
        self.labelFrame2.grid(column=0, row=4, padx=5, pady=5)

        self.labelFrame3 = ttk.LabelFrame(self, text='Image')
        self.labelFrame3.grid(column=0, row=5, padx=5, pady=5)

        self.labelFrame4 = ttk.LabelFrame(self, text='Run')
        self.labelFrame4.grid(column=0, row=6, padx=5, pady=5)

        self.button()

    def button(self):
        self.button = ttk.Button(
            self.labelFrame, text='Browse File', width=50, command=self.fileDialog)
        self.button.grid(column=0, row=1)

        self.spin = ttk.Spinbox(self.labelFrame2, from_=0, to=10, width=48)
        self.spin.grid(column=0, row=3)

        self.button1 = ttk.Button(
            self.labelFrame4, text='Run Program', width=50, command=self.RunPro)
        self.button1.grid(column=0, row=1)

    def RunPro(self):

        myimage = self.path
        mycluster = int(self.spin.get())  # get cluster from spnbox in tkinter

        # load the image and convert it from BGR to RGB
        image = cv2.imread(myimage)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # reshape the image to be a list of pixels
        image = image.reshape((image.shape[0] * image.shape[1], 3))

        # cluster the pixel intensities
        labels, centroids = kmeans(image, n_clusters=mycluster)

        # build a histogram of clusters and then create a figure
        # representing the number of pixels labeled to each color
        hist = centroid_histogram(labels)
        bar = plot_colors(hist, centroids)

        # show the color bar
        plt.figure()
        plt.axis("off")
        plt.imshow(bar)
        plt.show()

    def fileDialog(self):
        self.filename = filedialog.askopenfilename(initialdir='/', title='Select File',
                                                   filetype=(('jpeg', '*.jpg'), ('All Files', '*.*')))

        self.e1 = ttk.Entry(self.labelFrame1, width=50)
        self.e1.insert(0, self.filename)
        self.e1.grid(row=2, column=0, columnspan=50)

        Root.OpenImage(self.filename)

        newpath = self.filename
        self.path = newpath.replace('/', '\\\\')
        print(self.path)

        im = Image.open(self.path)
        resized = im.resize((300, 300), Image.ANTIALIAS)
        tkimage = ImageTk.PhotoImage(resized)
        myvar = ttk.Label(self.labelFrame3, image=tkimage)
        myvar.image = tkimage
        myvar.grid(column=0, row=4)

    def OpenImage(self):
        pass


def kmeans(X, n_clusters, max_iters=100):
    centroids = X[np.random.choice(
        range(X.shape[0]), size=n_clusters, replace=False)]

    for _ in range(max_iters):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([X[labels == k].mean(axis=0)
                                 for k in range(n_clusters)])
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

    return labels, centroids


def centroid_histogram(labels):
    numLabels = np.arange(0, len(np.unique(labels)) + 1)
    (hist, _) = np.histogram(labels, bins=numLabels)
    hist = hist.astype("float")
    hist /= hist.sum()
    return hist


def plot_colors(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0
    for (percent, color) in zip(hist, centroids):
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX
    return bar



root = Root()
root.mainloop()
