from w2v_lstm_functions import prepare_data, train_function, lstm_attn, plot_metrics
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import torch
import torch.nn as nn

drug = 'Pyrazinamide'
fld = '5'
print(drug, fld)

epochs = 50
m_name = '2'

lr = 0.001
momentum = 0.9
gamma = 0.95

train_vectors, y_train, test_vectors, y_test = prepare_data(drug, fld)
print('vectors: made')

hidden_dim = 16
bidir = False
model = lstm_attn(drug, fld, 100, hidden_dim, num_layers=1, bidirectional=bidir)

print('model: created')
print('hidden_dim:', hidden_dim, 'bidirectional:', bidir)

# weight of negative class
const = 0.003
w = y_train.sum().item() / y_train.shape[0] * const

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
#optimizer = optim.Adam(model.parameters(), lr=0.01)  
# weight_decay (float, optional) – weight decay (L2 penalty) (default: 0)
loss_fn = nn.CrossEntropyLoss(weight=torch.Tensor([w, 1-w]))
scheduler = ExponentialLR(optimizer, gamma=gamma)
#scheduler = None
print('model: training')
print('epochs:', epochs, 'weights:', f'{w:.5}, x{const}', 'lr:', lr, 'optim:', 'SGD', 'momentum:', momentum, 'scheduler:', 'exp', 'gamma:', gamma)
metrics = train_function(model, loss_fn, optimizer, train_vectors, y_train, test_vectors, y_test, m_name, epochs=epochs,
                         scheduler=scheduler, attn=True, save=True)

print('model: trained')
# save model
torch.save(model.state_dict(), f'lstm_models/{drug}_{fld}_model_{m_name}.pt')
plot_metrics(metrics, drug, 100, f'Attention: model {m_name}, fold{fld}', f'plots/{drug}_{fld}_model_{m_name}.jpeg')
print('plot: saved')