import time
import visdom
import torch
import logging
from torch.autograd import Variable

class Experiment(object):
    def __init__(self, model, viz_logging=False, name='pytorch_experiment', silent=False,
                 use_gpu=True):
        '''
        params
        model: Pytorch model that extends nn.Module
        viz_logging: (False) whether to use Visdom for logging in the experiment
        name: ("pytorch_experiment") what to call the environment in Visdom
        silent: (False) if true does not print to STDOUT during experiment, otherwise logs
        use_gpu: (True) will attempt to use GPU in available. If not available, will log warn and continue
        '''
        self._init_logging()
        
        self.model = model
        self.silent = silent
        self.viz = None
        self.log_interval = 2
        self._start_time = time.time()
        
        if use_gpu:
            self._use_gpu = torch.cuda.is_available()
            if self._use_gpu == False:
                self.log.warn('use_gpu is true but no gpu is detected')
            else:
                # GPU confirmed present, send model to GPU
                self.model = model.cuda()
            
        if viz_logging:
            self.viz = visdom.Visdom()
            self._viz_cache = {}

    def compile(self, optimizer, criterion):
        '''
        Sets the optimizer and criterion (loss function)
        params
        optimizer: an optimization function that extends optim.Optimizer
        criterion: a loss function
        '''
        
        self.optimizer = optimizer
        self.criterion = criterion
        if self._use_gpu:
            self.criterion = self.criterion.cuda()

    def fit(self, train_loader, n_epoch=10, valid_loader=None, valid_freq=5):
        '''
        Trains the experiment
        params
        train_loader: Dataloader for training
        n_epoch: (10) number of epochs to train for
        valid_loader: (None) a set to perform validation on
        valid_freq: (5) how often to test the validation set if one is provided
        '''
        for epoch in range(n_epoch):
            self._train(epoch, train_loader)
            
            if valid_loader:
                if epoch % valid_freq == 0:
                    self._valid(epoch, valid_loader)
    
    def evaluate(self, test_loader):
        self._test(test_loader)
    
    
    def _train(self, epoch, dataloader):
        '''
        _train performs one epoch of training
        '''
        self.model.train()
        total_loss = 0.0
        
        for (X, y) in dataloader:
            X = Variable(X)
            y = Variable(y)

            if self._use_gpu:
                X = X.cuda()
                y = y.cuda()

            y_hat = self.model(X)
            loss = self.criterion(y_hat, y)
            total_loss += loss.data[0]

            # backprop error, update weights, zero old grads
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        avg_loss = total_loss / len(dataloader.dataset)
        self._log_train_loss(epoch, avg_loss)
        
    
    def _log_train_loss(self, epoch, avg_loss):
        if epoch % 2 != 0:
            return
        if not self.silent:
            print('TRAIN epoch: {} avg_loss: {}'.format(epoch, avg_loss))
        epoch, avg_loss = torch.Tensor([epoch]), torch.Tensor([avg_loss])
        if self.viz:
            self._log_vis_line('train_avg_loss', epoch, avg_loss)
    
    def _valid(self, epoch, dataloader):
        '''
        Performs one epoch on validation data and logs
        '''
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        correct = 0.0        

        for (X, y) in dataloader:
            # our model is expecting batches of 1D inputs
            X = Variable(X.float())
            y = Variable(y)

            if self._use_gpu:
                X = X.cuda()
                y = y.cuda()

            y_hat = self.model(X)
            loss = self.criterion(y_hat, y)

            # metrics
            total_loss += loss.data[0]
#             total_acc += categorical_accuracy(y_hat, y)
            m = y_hat.data.max(1)[1]
            correct += (m == y.data).sum()

        avg_loss = total_loss / len(dataloader.dataset)
        avg_accuracy = correct / len(dataloader.dataset)
        self._log_valid(epoch, avg_accuracy, avg_loss)

    def _log_valid(self, epoch, avg_acc, avg_loss):
        if not self.silent:
            print('VALIDATION: Epoch: {} \nAvg loss: {:.4f}, Accuracy: {}'.format(
                epoch, avg_loss, avg_acc
            ))
        if self.viz:
            epoch, avg_acc, avg_loss = torch.Tensor([epoch]), torch.Tensor([avg_acc]), torch.Tensor([avg_loss])
            self._log_vis_line('valid_avg_loss', epoch, avg_loss)
            self._log_vis_line('valid_avg_acc', epoch, avg_acc)

    def _test(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        correct = 0.0

        for (X, y) in dataloader:
            # our model is expecting batches of 1D inputs
            X = Variable(X.float())
            y = Variable(y)

            if self._use_gpu:
                X = X.cuda()
                y = y.cuda()

            y_hat = self.model(X)
            loss = self.criterion(y_hat, y)

            # metrics
            total_loss += loss.data[0]
            m = y_hat.data.max(1)[1]
            correct += (m == y.data).sum()

        avg_loss = total_loss / len(dataloader.dataset)
        avg_accuracy = correct / len(dataloader.dataset)
        self._log_test(0, avg_accuracy, avg_loss)
    
    def _log_test(self, epoch, avg_acc, avg_loss):
        if not self.silent:
            print('TEST: Avg loss: {:.4f}, Accuracy: {}'.format(avg_loss, avg_acc))
        if self.viz:
            epoch, avg_acc, avg_loss = torch.Tensor([epoch]), torch.Tensor([avg_acc]), torch.Tensor([avg_loss])
            self._log_vis_line('test_avg_loss', epoch, avg_loss)
            self._log_vis_line('test_avg_acc', epoch, avg_acc)

    def _log_vis_line(self, metric_name, x, y):
        if metric_name in self._viz_cache:
            self._viz_cache[metric_name] = self.viz.line(
                X=x,Y=y,
                win=self._viz_cache[metric_name], update='append', opts={'title': self._make_title(metric_name)})
        else:
            self._viz_cache[metric_name] = self.viz.line(X=x, Y=y, opts={'title': self._make_title(metric_name)})
    
    def _make_title(self, metric_name):
        return metric_name + "_" + str(self._start_time)
    
    def _init_logging(self):
        self.log = logging.getLogger('PyTorchExperiment')
        self.log.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s/%(name)s [%(levelname)s]: %(message)s')
        handler = logging.StreamHandler()
        self.log.addHandler(handler)


def categorical_accuracy(y_pred, y_true):
    '''Calculates categorical accuracy for a batch of predictions and labels'''
    m = y_pred.data.max(1)[1]
    correct = (m == y_true.data).sum()
    return correct / float(len(y_true))
    