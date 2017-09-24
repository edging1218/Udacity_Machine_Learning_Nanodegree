import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.mixture import GaussianMixture
import pickle
import random
from numpy import linalg
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt


class Data:
    def __init__(self):
        self.data_path = ['input/Crimes_-_'+str(i)+'.csv' for i in range(2015, 2017)]
        self.target_name = 'Primary Type'
        self.to_delete = ['ID', 'Case Number', 'Block', 'IUCR', 'FBI Code',
                          'Description', 'X Coordinate', 'Y Coordinate',
                          'Updated On', 'Location']
        self.data = None
        self.target = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.community_name = pickle.load(open('input/community_name.p'))

        self.read_data()
        self.extract_features()

    def read_data(self):
        """
        Read in train and test data
        """
        print 'Read in data...'
        dataset = []
        for path in self.data_path:
            dataset.append(pd.read_csv(path,
                                       # nrows=5000,
                                       index_col='Date',
                                       parse_dates=['Date']))
        self.data = pd.concat(dataset, axis=0)
        self.data = self.data.dropna(axis=0, how='any')
        print self.data.info()

    def random_sample(self, sample_size):
        """
        Randomly sample the data with input sampled data size
        """
        index = random.sample(np.arange(self.data.shape[0]), int(self.data.shape[0] * sample_size))
        self.data = self.data.iloc[index, :]

    def dummies(self, col, name):
        """
        Create dummies
        """
        series = self.data[col]
        del self.data[col]
        dummies = pd.get_dummies(series, prefix=name)
        self.data = pd.concat([self.data, dummies], axis=1)

    def label_encoder(self, col):
        """
        Create encoded label for input column
        """
        le = LabelEncoder()
        self.data[col] = le.fit_transform(self.data[col])

    def keep_major_cat(self, col, top_n):
        """
        Keep only major category
        """
        counts = self.data[col].value_counts()

        def location(x):
            if x == 'APARTMENT':
                return 'RESIDENCE'
            if x in counts.index[:top_n]:
                return x
            else:
                return 'OTHER'

        def primary_type(x):
            if x in counts.index[:top_n] or x == 'HOMICIDE':
                return x
            else:
                return 'OTHER OFFENSE'

        if col == 'Primary Type':
            self.target_name = counts.index[:top_n].tolist() 
            self.target_name.append('HOMICIDE')
            self.data[col] = self.data[col].apply(lambda x: primary_type(x))
        elif col == 'Location Description':
            self.data[col] = self.data[col].apply(lambda x: location(x))

    def extract_features(self):
        """
        Extract new features from original features
        """
        print '\nExtract features...'
        self.data = self.data.drop(self.to_delete, axis=1)
        self.data['Hour'] = self.data.index.hour
        self.data['Weekday'] = self.data.index.weekday
        self.data['Month'] = self.data.index.month
        # self.keep_major_cat('Location Description', 10)
        self.keep_major_cat('Primary Type', 8)
        print self.data.info()

    def create_cluster(self):
        gmm = GaussianMixture(n_components=50, covariance_type='full', max_iter=300, n_init=2).\
            fit(self.data[['Latitude', 'Longitude']])
        self.data['Cluster'] = gmm.predict(self.data[['Latitude', 'Longitude']])
        # print self.data['Cluster'].value_counts().sort_index()
        # Plot cluster
        means, covars = gmm.means_, gmm.covariances_
        fig, ax = plt.subplots(figsize=(8, 6))
        for mean, covar in zip(means, covars):
            v, w = linalg.eigh(covar)
            v = 2. * np.sqrt(2.) * np.sqrt(v)
            u = w[0] / linalg.norm(w[0])
            # Plot an ellipse to show the Gaussian component
            angle = np.arctan(u[1] / u[0])
            angle = 180. * angle / np.pi  # convert to degrees
            ell = Ellipse(mean, v[0], v[1], 180. + angle)
            # ell.set_clip_box(ax.bbox)
            ell.set_alpha(0.5)
            ax.add_patch(ell)
        urcornerlat = 42.03
        urcornerlong = -87.51
        llcornerlat = 41.64
        llcornerlong = -87.95
        plt.axis([llcornerlat, urcornerlat, llcornerlong, urcornerlong])
        plt.xlabel('Latitude', fontsize=18)
        plt.ylabel('Longitude', fontsize=18)
        plt.show()

    def preprocessing(self):
        """
        Wrap-up function for create_dummies, label_encoder and split data
        """
        self.data.index = np.arange(self.data.shape[0])
        # self.dummies('Hour', 'Hour_')
        # self.dummies('Weekday', 'Day_')
        # self.dummies('Month', 'Month_')
        # self.dummies('Location Description', 'Loc_')
        # self.dummies('Domestic', 'Domestic_')
        # self.target_name = [x for x in self.data.columns if x[:4] == 'Type']
        self.label_encoder('Location Description')
        # self.label_encoder('Primary Type')
        self.create_cluster()
        # print self.data.info()
        self.split_data()

    def split_data(self):
        """
        Split data into train and test set with ratio 6:4
        """
        print 'Splitting data...'
        self.target = self.data['Primary Type']
        del self.data['Primary Type']
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.data,
                                                                                self.target,
                                                                                test_size=0.4,
                                                                                random_state=1)
        print 'x_Training set has {} rows, {} columns.'.format(*self.x_train.shape)
        print 'x_Test set has {} rows, {} columns.'.format(*self.x_test.shape)

    def data_info(self):
        """
        Info of train and test data
        """
        print '\nTrain:\n{}\n'.format('-' * 50)
        self.x_train.info()
        print '\nTrain target:\n{}\n'.format('-' * 50)
        self.y_train.info()

    def data_peek(self):
        """
        Peek at the train and test data
        """
        print '\nTrain:\n{}\n'.format('-'*50)
        print self.x_train.head()
        print '\nTrain target:\n{}\n'.format('-'*50)
        print self.y_train.head()


