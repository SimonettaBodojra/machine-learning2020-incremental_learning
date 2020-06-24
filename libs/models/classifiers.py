import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

# [FCClassifier] simply classifies according to the output of the network by choosing the most likable class
class FCClassifier():
  def __init__(self):
      self.net = None

  def update(self, step, net, train_dataloader):
      self.net = net

  def classify(self, input_images):
    self.net = self.net.cuda()
    self.net.train(False)
    with torch.no_grad():
      output = self.net(input_images, output = 'fc')
      preds = torch.argmax(output, dim = 1).cuda()
      return preds

    import torch
    from sklearn.model_selection import GridSearchCV
    from sklearn.neighbors import KNeighborsClassifier
    from imblearn.under_sampling import RandomUnderSampler

    # classifier using the NN policy
    # [k_values]: the n_neighbors value to be tried during the grid search
    class KNNClassifier():
        def __init__(self, k_values=[9, 11, 13, 15]):
            self.net = None
            self.k_param_grid = {"n_neighbors": k_values}
            self.classifier = KNeighborsClassifier(weights="distance")

        def update(self, step, net, train_dataloader):
            self.net = net
            images_tot = None
            labels_tot = None
            self.net = self.net.cuda()
            self.net.train(False)
            with torch.no_grad():
                for images, labels in train_dataloader:
                    if images_tot is None:
                        images_tot = images
                        labels_tot = labels
                    else:
                        # Take all the images and labels to be fitted by the classifier
                        images_tot = torch.cat((images_tot, images), 0)
                        labels_tot = torch.cat((labels_tot, labels), 0)

                images_tot = images_tot.cuda()
                features = self.net(images_tot, output='features')
                features = features.cpu()

                # random undersampling to tackle the unbalance between the new data and old examplars
                # it is like simulating the random selection of the exemplars
                rus = RandomUnderSampler()
                features, labels_tot = rus.fit_resample(features, labels_tot)
                # selecting the best K through grid search with cross-validation
                gs = GridSearchCV(estimator=self.classifier, param_grid=self.k_param_grid, cv=4)
                gs.fit(features, labels_tot)
                self.classifier = gs.best_estimator_

        def classify(self, images):
            preds = []
            self.net = self.net.cuda()
            self.net.train(False)
            with torch.no_grad():
                features = self.net(images, output='features')
                features = features.cpu()
                preds = self.classifier.predict(features)
            return torch.Tensor(preds).cuda()

        class BiasLayer(nn.Module):
            """
            Bias Correction Layer with 2 parameters
            """

            def __init__(self):
                super(BiasLayer, self).__init__()
                self.alpha = nn.Parameter(torch.ones(1, requires_grad=True, device="cuda"))
                self.beta = nn.Parameter(torch.zeros(1, requires_grad=True, device="cuda"))

            def forward(self, x):
                return self.alpha * x + self.beta

            def printParam(self):
                print(self.alpha.item(), self.beta.item())