import copy
import json
import os
import pprint
import sys
import torch
import torch.nn as nn
import torch.optim as optim

from data_utils import load_train_and_test_data, get_jth_minibatach
from metrics import print_and_ret_dnn_metrics


class NeuralNet(nn.Module):
    
    '''Fully connected neural network with one hidden layer
    '''
    
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

    


print('[DOING] Load and parse config')
# load config
# config_path = './config.json'
config_path = sys.argv[1]
with open(config_path) as f:
    config = json.load(f)
pprint.pprint(config)

# parse parameters
train_data_path = config['train_data_path']
test_data_path = config['test_data_path']
train_data_num = config['train_data_num']
class_num = config['class_num']
random_state = config['random_state']
pre_n_bytes = config['pre_n_bytes']
hidden_size = config['hidden_size']
learning_rate = config['learning_rate']
epoch_nums = config['epoch_nums']
iters_per_epoch = config['iters_per_epoch']
batch_size = config['batch_size']
use_gpu = config['use_gpu']
result_save_path = config['result_save_path']

os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(use_gpu)
device = torch.device('cuda:0' if torch.cuda.is_available() and len(use_gpu) > 0 else 'cpu')
if not os.path.exists(result_save_path):
    os.makedirs(result_save_path)
print('[SUCCESS] Load and parse config\n\n')


print('[DOING] Load data')
# load data
X_train, y_train, X_test, y_test = load_train_and_test_data(train_data_path, class_num, 
                                                            train_data_num, random_state, 
                                                            test_data_path, pre_n_bytes,
                                                            )
print('[SUCCESS] Load data\n\n')


print('[DOING] Train DNN')
# train DNN
model = NeuralNet(pre_n_bytes, hidden_size, class_num)
if torch.cuda.device_count() > 1:
    print('Use', torch.cuda.device_count(), 'GPUs!')
    model = nn.DataParallel(model)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
min_loss = 999999999
min_model_state_dict = None

for i in range(epoch_nums):
    # train DNN model
    print('Training DNN model Epoch [{0}/{1}]'.format(i, epoch_nums))
    for j in range(iters_per_epoch):
        trn_x, trn_y = get_jth_minibatach(j, batch_size, X_train, y_train)
        trn_x = trn_x.to(device)
        trn_y = trn_y.to(device)
        output = model(trn_x)
        loss = criterion(output, trn_y)
        if loss < min_loss:
            min_model_state_dict = copy.deepcopy(model.state_dict())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if j % 100 == 0:
            print('Epoch: {0} Iter: {1} loss: {2:.4f}'.format(i, j, loss.item()))
    if i % 10 == 0:
        with torch.no_grad():
            X_test = X_test.to(device)
            y_test = y_test.to(device)
            outputs = model(X_test)
            _, predicted = torch.max(outputs.data, 1)

            y_true = y_test.cpu().numpy()
            y_pred = predicted.cpu().numpy()
            print_and_ret_dnn_metrics(y_true, y_pred)
            print('\nBenign 1 DDoS 0')
            y_true_xor = y_true ^ 1
            y_pred_xor = y_pred ^ 1
            print_and_ret_dnn_metrics(y_true_xor, y_pred_xor)
            torch.save(model.state_dict(), os.path.join(result_save_path, 'model_dnn_checkpoints_' + str(i) + '.pth'))
torch.save(min_model_state_dict, os.path.join(result_save_path, 'model_dnn.pth'))
print('[SUCCESS] Train DNN')


print('[DOING] Test DNN')
model.load_state_dict(torch.load(os.path.join(result_save_path, 'model_dnn.pth')))
with torch.no_grad():
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    outputs = model(X_test)
    _, predicted = torch.max(outputs.data, 1)

    y_true = y_test.cpu().numpy()
    y_pred = predicted.cpu().numpy()
    print_and_ret_dnn_metrics(y_true, y_pred)
    print('\nBenign 1 DDoS 0')
    y_true_xor = y_true ^ 1
    y_pred_xor = y_pred ^ 1
    print_and_ret_dnn_metrics(y_true_xor, y_pred_xor)
print('[SUCCESS] Test DNN')