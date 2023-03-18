import copy
import joblib
import json
import os
import pprint
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import sys
import torch
import torch.nn as nn
import torch.optim as optim

from data_utils import load_train_and_test_data, get_jth_minibatach
from metrics import print_and_ret_all_metrics, ret_all_metrics
from visualize_utils import visualize_decision_tree

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

class SurrogateModel(nn.Module):
    
    '''Fully connected neural network with one hidden layer
       Split the fc1 into two parts 
       because only in this way can have compute graph with DNN model weights
       so that can backpropagation to update DNN model weights and this is tree regularization
       (maybe have other ways to do this faster. Currently this is not very elegant.)
    '''
    
    def __init__(self, input_size, hidden_size, class_num):
        super(SurrogateModel, self).__init__()
        self.fc1_1 = nn.Linear(input_size*hidden_size, 32)
        self.fc1_2 = nn.Linear(hidden_size*class_num, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)
    
    def forward(self, x):
        # x is the model.state_dict().items()[training] or model.named_parameters()[calculate APL]
        for key, value in x:
            if key.endswith('fc1.weight'):
                out1 = self.fc1_1(value.view(-1))
            elif key.endswith('fc2.weight'):
                out2 = self.fc1_2(value.view(-1))
        out = out1 + out2
        out = self.relu(out)
        out = self.fc2(out)
        
        return out

def get_y_surrogate_train(saved_model_state_dict, X_train, y_train, pre_n_bytes, hidden_size, class_num, min_samples_leaf, tree_model, tree_num, lambda_APL, lambda_acc, lambda_pre, lambda_rec, lambda_f1):
    tmp_model = NeuralNet(pre_n_bytes, hidden_size, class_num)
    if torch.cuda.device_count() > 1:
        print('Use', torch.cuda.device_count(), 'GPUs!')
        tmp_model = nn.DataParallel(tmp_model)
    tmp_model.to(device)
    y_APL_train = torch.zeros(len(saved_model_state_dict))
    y_acc_train = torch.zeros(len(saved_model_state_dict))
    y_pre_train = torch.zeros(len(saved_model_state_dict))
    y_rec_train = torch.zeros(len(saved_model_state_dict))
    y_f1_train = torch.zeros(len(saved_model_state_dict))
    # save the best tree during this iteration
    ibest_tree_acc = 0
    ibest_tree_f1 = 0
    ibest_tree_apl = 0
    ibest_i = 0
    # save the min loss tree during this iteration
    iloss = 1e9
    iloss_tree_acc = 999999999
    iloss_tree_f1 = 999999999
    iloss_tree_apl = 999999999
    iloss_i = 0
    for i in range(len(saved_model_state_dict)):
        tmp_model.load_state_dict(saved_model_state_dict[i])
        X_train = X_train.to(device)
        outputs = tmp_model(X_train)
        _, y_pred = torch.max(outputs.data, 1)
        if tree_model == 'decisionTree':
            tree = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf)
            tree.fit(X_train.cpu().numpy(), y_pred.cpu().numpy())
        elif tree_model == 'randomForest':
            tree = RandomForestClassifier(n_estimators=tree_num, min_samples_leaf=min_samples_leaf)
            tree.fit(X_train.cpu().numpy(), y_pred.cpu().numpy())
        # calculate acc precision recall f1 score Benign 1 DDoS 0
        y_train_xor = y_train ^ 1
        y_tree_pred = tree.predict(X_train.cpu().numpy())
        y_tree_pred_xor = y_tree_pred ^ 1
        acc, pre, rec, f1, apl, leaf_num = ret_all_metrics(y_train_xor, y_tree_pred_xor, X_train.cpu().numpy(), tree)
        y_acc_train[i] = acc
        y_pre_train[i] = pre
        y_rec_train[i] = rec
        y_f1_train[i] = f1
        # update best tree
        if f1 > ibest_tree_f1:
            ibest_tree_f1 = f1
            ibest_tree_acc = acc
            ibest_tree_apl = apl
            ibest_i = i
        elif abs(f1 - ibest_tree_f1) < 1e-4:
            if apl < ibest_tree_apl:
                ibest_tree_f1 = f1
                ibest_tree_acc = acc
                ibest_tree_apl = apl
                ibest_i = i
        # update min loss tree
        loss = lambda_APL * apl - lambda_acc * acc - lambda_pre * pre - lambda_rec * rec - lambda_f1 * f1
        if loss < iloss:
            iloss = loss
            iloss_tree_acc = acc
            iloss_tree_f1 = f1
            iloss_tree_apl = apl
            iloss_i = i
    return y_APL_train, y_acc_train, y_pre_train, y_rec_train, y_f1_train, ibest_tree_acc, ibest_tree_f1, ibest_tree_apl, ibest_i, iloss, iloss_tree_acc, iloss_tree_f1, iloss_tree_apl, iloss_i

    


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
batch_size_surrogate = config['batch_size_surrogate']
tree_model = config['tree_model']
tree_num = config['tree_num']
min_samples_leaf = config['min_samples_leaf']
use_gpu = config['use_gpu']
lambda_APL = config['lambda_APL']
lambda_acc = config['lambda_acc']
lambda_pre = config['lambda_pre']
lambda_rec = config['lambda_rec']
lambda_f1 = config['lambda_f1']
epsilon_punish = config["epsilon_punish"]
class_names = config['class_names']
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
# train DNN with tree regularization
model = NeuralNet(pre_n_bytes, hidden_size, class_num)
if torch.cuda.device_count() > 1:
    print('Use', torch.cuda.device_count(), 'GPUs!')
    model = nn.DataParallel(model)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# surrogate model APL
surrogate_model_APL = SurrogateModel(pre_n_bytes, hidden_size, class_num)
surrogate_model_APL.to(device)
criterion_surrogate = nn.MSELoss()
optimizer_surrogate_apl = optim.Adam(surrogate_model_APL.parameters(), lr=learning_rate)

# surrogate model acc score
surrogate_model_acc = SurrogateModel(pre_n_bytes, hidden_size, class_num)
surrogate_model_acc.to(device)
optimizer_surrogate_acc = optim.Adam(surrogate_model_acc.parameters(), lr=learning_rate)

# surrogate model precision score
surrogate_model_precision = SurrogateModel(pre_n_bytes, hidden_size, class_num)
surrogate_model_precision.to(device)
optimizer_surrogate_precision = optim.Adam(surrogate_model_precision.parameters(), lr=learning_rate)

# surrogate model recall score
surrogate_model_recall = SurrogateModel(pre_n_bytes, hidden_size, class_num)
surrogate_model_recall.to(device)
optimizer_surrogate_recall = optim.Adam(surrogate_model_recall.parameters(), lr=learning_rate)

# surrogate model f1 score
surrogate_model_f1 = SurrogateModel(pre_n_bytes, hidden_size, class_num)
surrogate_model_f1.to(device)
optimizer_surrogate_f1 = optim.Adam(surrogate_model_f1.parameters(), lr=learning_rate)

# save best model weights
best_tree_acc = 0
best_tree_f1 = 0
best_tree_apl = 0
best_model_weights = None

# save min loss model weights
min_tree_loss = 1e9
min_tree_acc = 0
min_tree_f1 = 0
min_tree_apl = 0
min_model_weights = None

for i in range(epoch_nums):
    print('*' * 20)
    if i == 0 or i % 10 == 0:
        saved_model_state_dict = [] # save the model state dict in each ten epoch
    # train DNN model
    print('Training DNN model......')
    for j in range(iters_per_epoch):
        trn_x, trn_y = get_jth_minibatach(j, batch_size, X_train, y_train)
        trn_x = trn_x.to(device)
        trn_y = trn_y.to(device)
        output = model(trn_x)
        tree_apl = surrogate_model_APL(model.named_parameters())
        tree_acc = surrogate_model_acc(model.named_parameters())
        tree_precision = surrogate_model_precision(model.named_parameters())
        tree_recall = surrogate_model_recall(model.named_parameters())
        tree_f1 = surrogate_model_f1(model.named_parameters())
        if i < 10:
            loss = criterion(output, trn_y)
        else:
            loss = criterion(output, trn_y) + lambda_APL * tree_apl - lambda_acc * tree_acc - lambda_pre * tree_precision - lambda_rec * tree_recall - lambda_f1 * tree_f1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        saved_model_state_dict.append(copy.deepcopy(model.state_dict()))
        if j % 100 == 0:
            print('Epoch: {0} Iter: {1} Estimated APL: {2} Estimated Acc: {3} Estimated: F1: {4} Loss: {5}'.format(i, j, tree_apl.item(), tree_acc.item(), tree_f1.item(), loss.item()))
    # train Decision Tree to get {weights, APL/f1} dataset
    print('Getting {weights, y_surrogate_train} dataset......')
    y_APL_train, y_acc_train, y_pre_train, y_rec_train, y_f1_train, ibest_tree_acc, ibest_tree_f1, ibest_tree_apl, ibest_i, iloss, iloss_tree_acc, iloss_tree_f1, iloss_tree_apl, iloss_i = get_y_surrogate_train(saved_model_state_dict, X_train, y_train, pre_n_bytes, hidden_size, class_num, min_samples_leaf, tree_model, tree_num, lambda_APL, lambda_acc, lambda_pre, lambda_rec, lambda_f1)
    print('Mean Acc: {0:.4f}'.format(y_acc_train.mean().item()))
    print('Mean APL: {0:.4f}'.format(y_APL_train.mean().item()))
    print('Mean f1: {0:.4f}'.format(y_f1_train.mean().item()))
    print('Iter Best Tree accuracy: {0:.4f}'.format(ibest_tree_acc))
    print('Iter Best Tree f1: {0:.4f}'.format(ibest_tree_f1))
    print('Iter Best Tree APL: {0:.4f}'.format(ibest_tree_apl))
    if ibest_tree_f1 > best_tree_f1:
        best_tree_acc = ibest_tree_acc
        best_tree_f1 = ibest_tree_f1
        best_tree_apl = ibest_tree_apl
        best_model_weights = copy.deepcopy(saved_model_state_dict[ibest_i])
    elif abs(ibest_tree_f1 - best_tree_f1) < 1e-4:
        if ibest_tree_apl < best_tree_apl:
            best_tree_acc = ibest_tree_acc
            best_tree_f1 = ibest_tree_f1
            best_tree_apl = ibest_tree_apl
            best_model_weights = copy.deepcopy(saved_model_state_dict[ibest_i])
    print('All Best Tree accuracy: {0:.4f}'.format(best_tree_acc))
    print('All Best Tree f1: {0:.4f}'.format(best_tree_f1))
    print('All Best Tree APL: {0:.4f}'.format(best_tree_apl))
    print('Iter Min Loss: {0:.4f}'.format(iloss))
    print('Iter Min Loss Tree accuracy: {0:.4f}'.format(iloss_tree_acc))
    print('Iter Min Loss Tree f1: {0:.4f}'.format(iloss_tree_f1))
    print('Iter Min Loss Tree apl: {0:.4f}'.format(iloss_tree_apl))
    if iloss < min_tree_loss:
        min_tree_loss = iloss
        min_tree_acc = iloss_tree_acc
        min_tree_f1 = iloss_tree_f1
        min_tree_apl = iloss_tree_apl
        min_model_weights = copy.deepcopy(saved_model_state_dict[iloss_i])
    print('All Min Loss: {0:.4f}'.format(min_tree_loss))
    print('All Min Loss Tree accuracy: {0:.4f}'.format(min_tree_acc))
    print('All Min Loss Tree f1: {0:.4f}'.format(min_tree_f1))
    print('All Min Loss Tree apl: {0:.4f}'.format(min_tree_apl))
    print('Training surrogate model......')
    # train surrogate model
    for j in range(1000):
        trn_x_apl, trn_y_apl = get_jth_minibatach(j, batch_size_surrogate, saved_model_state_dict, y_APL_train)
        trn_x_acc, trn_y_acc = get_jth_minibatach(j, batch_size_surrogate, saved_model_state_dict, y_acc_train)
        trn_x_precision, trn_y_precision = get_jth_minibatach(j, batch_size_surrogate, saved_model_state_dict, y_pre_train)
        trn_x_recall, trn_y_recall = get_jth_minibatach(j, batch_size_surrogate, saved_model_state_dict, y_rec_train)
        trn_x_f1, trn_y_f1 = get_jth_minibatach(j, batch_size_surrogate, saved_model_state_dict, y_f1_train)
        trn_y_apl = trn_y_apl.to(device)
        trn_y_acc = trn_y_acc.to(device)
        trn_y_precision = trn_y_precision.to(device)
        trn_y_recall = trn_y_recall.to(device)
        trn_y_f1 = trn_y_f1.to(device)
        output_apl = torch.zeros(trn_y_apl.size(0), device=device)
        output_acc = torch.zeros(trn_y_acc.size(0), device=device)
        output_precision = torch.zeros(trn_y_precision.size(0), device=device)
        output_recall = torch.zeros(trn_y_recall.size(0), device=device)
        output_f1 = torch.zeros(trn_y_f1.size(0), device=device)
        for k in range(len(trn_x_apl)):
            output_apl[k] = surrogate_model_APL(trn_x_apl[k].items())
            output_acc[k] = surrogate_model_acc(trn_x_acc[k].items())
            output_precision[k] = surrogate_model_precision(trn_x_precision[k].items())
            output_recall[k] = surrogate_model_recall(trn_x_recall[k].items())
            output_f1[k] = surrogate_model_f1(trn_x_f1[k].items())
        loss_apl = criterion_surrogate(output_apl, trn_y_apl)
        loss_acc = criterion_surrogate(output_acc, trn_y_acc)
        loss_precision = criterion_surrogate(output_precision, trn_y_precision)
        loss_recall = criterion_surrogate(output_recall, trn_y_recall)
        loss_f1 = criterion_surrogate(output_f1, trn_y_f1)
        # l2 norm
        l2_norm_apl = 0
        for key, value in surrogate_model_APL.named_parameters():
            if key.endswith('weight'):
                l2_norm_apl += value.norm()
        loss_apl += epsilon_punish * l2_norm_apl
        l2_norm_acc = 0
        for key, value in surrogate_model_acc.named_parameters():
            if key.endswith('weight'):
                l2_norm_acc += value.norm()
        loss_acc += epsilon_punish * l2_norm_acc
        l2_norm_precision = 0
        for key, value in surrogate_model_precision.named_parameters():
            if key.endswith('weight'):
                l2_norm_precision += value.norm()
        loss_precision += epsilon_punish * l2_norm_precision
        l2_norm_recall = 0
        for key, value in surrogate_model_recall.named_parameters():
            if key.endswith('weight'):
                l2_norm_recall += value.norm()
        loss_recall += epsilon_punish * l2_norm_recall
        l2_norm_f1 = 0
        for key, value in surrogate_model_f1.named_parameters():
            if key.endswith('weight'):
                l2_norm_f1 += value.norm()
        loss_f1 += epsilon_punish * l2_norm_f1
        optimizer_surrogate_apl.zero_grad()
        optimizer_surrogate_acc.zero_grad()
        optimizer_surrogate_precision.zero_grad()
        optimizer_surrogate_recall.zero_grad()
        optimizer_surrogate_f1.zero_grad()
        loss_apl.backward()
        loss_acc.backward()
        loss_precision.backward()
        loss_recall.backward()
        loss_f1.backward()
        optimizer_surrogate_apl.step()
        optimizer_surrogate_acc.step()
        optimizer_surrogate_precision.step()
        optimizer_surrogate_recall.step()
        optimizer_surrogate_f1.step()
        if j % 200 == 0:
            print('Surrogate Iters: {0} APL loss: {1:.4f} f1 loss: {2:.4f}'.format(j, loss_apl.item(), loss_f1.item()))
    if i % 10 == 0:
        with torch.no_grad():
            tmp_model = NeuralNet(pre_n_bytes, hidden_size, class_num)
            if torch.cuda.device_count() > 1:
                print('Use', torch.cuda.device_count(), 'GPUs!')
                tmp_model = nn.DataParallel(tmp_model)
            tmp_model.to(device)
            tmp_model.load_state_dict(best_model_weights)
            X_train = X_train.to(device)
            outputs = tmp_model(X_train)
            _, y_pred = torch.max(outputs.data, 1)
            if tree_model == 'decisionTree':
                tree = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf)
                tree.fit(X_train.cpu().numpy(), y_pred.cpu().numpy())
            elif tree_model == 'randomForest':
                tree = RandomForestClassifier(n_estimators=tree_num, min_samples_leaf=min_samples_leaf)
                tree.fit(X_train.cpu().numpy(), y_pred.cpu().numpy())
            y_true = y_test.cpu().numpy()
            y_tree_pred = tree.predict(X_test.cpu().numpy())
            print_and_ret_all_metrics(y_true, y_tree_pred, X_test.cpu().numpy(), tree)
            print('Benign 1 DDoS 0')
            y_true_xor = y_true ^ 1
            y_tree_pred_xor = y_tree_pred ^ 1
            print_and_ret_all_metrics(y_true_xor, y_tree_pred_xor, X_test.cpu().numpy(), tree)
print('[SUCCESS] Train DNN\n\n')

# In[14]:


torch.save(best_model_weights, os.path.join(result_save_path, 'model_dnn_to_tree_best_dnn.pth'))
torch.save(min_model_weights, os.path.join(result_save_path, 'model_dnn_to_tree_minloss_dnn.pth'))


# In[16]:


# visualize & test
print('[DOING] DNN To Tree & Tree Visualize')
model = NeuralNet(pre_n_bytes, hidden_size, class_num)
if torch.cuda.device_count() > 1:
    print('Use', torch.cuda.device_count(), 'GPUs!')
    model = nn.DataParallel(model)
model.to(device)
model.load_state_dict(torch.load(os.path.join(result_save_path, 'model_dnn_to_tree_best_dnn.pth')))
X_train = X_train.to(device)
outputs = model(X_train)
_, y_pred = torch.max(outputs.data, 1)
if tree_model == 'decisionTree':
    tree = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf)
    tree.fit(X_train.cpu().numpy(), y_pred.cpu().numpy())
    joblib.dump(tree, os.path.join(result_save_path, 'model_dnn_to_tree_tree.joblib'))
elif tree_model == 'randomForest':
    tree = RandomForestClassifier(n_estimators=tree_num, min_samples_leaf=min_samples_leaf)
    tree.fit(X_train.cpu().numpy(), y_pred.cpu().numpy())
    joblib.dump(tree, os.path.join(result_save_path, 'model_dnn_to_tree_rf.joblib'))
y_true = y_test.cpu().numpy()
y_tree_pred = tree.predict(X_test.cpu().numpy())
print_and_ret_all_metrics(y_true, y_tree_pred, X_test.cpu().numpy(), tree)
print('\nBenign 1 DDoS 0')
y_true_xor = y_true ^ 1
y_tree_pred_xor = y_tree_pred ^ 1
print_and_ret_all_metrics(y_true_xor, y_tree_pred_xor, X_test.cpu().numpy(), tree)

if tree_model == 'decisionTree':
    visualize_decision_tree(tree, pre_n_bytes, class_names, result_save_path, 'visualize_tree.pdf')
elif tree_model == 'randomForest':
    for idx, t in enumerate(tree.estimators_):
        visualize_decision_tree(t, pre_n_bytes, class_names, result_save_path, 'visualize_tree' + str(idx) + '.pdf')
print('[SUCCESS] DNN To Tree & Tree Visualize\n\n')