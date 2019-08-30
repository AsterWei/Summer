import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision
from torch.autograd import Variable
from functions import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
import pickle

#set path
data_path='outpath/'
action_name='answers.txt'
save_model_path='result_crnn/'

# set cnn parameters
CNN_fc_hidden1, CNN_fc_hidden2=640,60
CNN_embed_dim=512 #latente dim extracted by 2DCNN
res_size=224 #ResNet image size
dropout_p=0.0

#DecoderRNN architecture
RNN_hidden_layers=3
RNN_hidden_nodes=512
RNN_FC_dim=256

#Training parametrs
k=2 #target category
trynum=20
epochs=8 
batch_size=5
learning_rate=0.01
log_interval=4
img_x, img_y=360,640
test_size=0.4

#select frame
begin_frame=5
end_frame=55
skip_frame=5


def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def train(log_interval, model, device, train_loader, optimizer, epoch):
    # set model as training mode
    cnn_encoder,rnn_decoder=model
    cnn_encoder.train()
    rnn_decoder.train()
    losses = []
    scores = []
    N_count = 0   # counting total trained sample in one epoch
    for batch_idx, (X, y) in enumerate(train_loader):
        # distribute data to device
        X, y = X.to(device), y.to(device).view(-1, )#X,y---input&output
        N_count += X.size(0)

        optimizer.zero_grad()
        print(X.shape)
        output = rnn_decoder(cnn_encoder(X))  # output size = (batch, number of classes)
        print(output)
        loss = F.cross_entropy(output, y)
        losses.append(loss.item())
        # compute accuracy
        y_pred = torch.max(output, 1)[1] # y_pred != output
        print(y_pred.cpu().numpy().shape)
        print(y.cpu().numpy().shape)
        step_score = accuracy_score(y.cpu().numpy(), y_pred.cpu().numpy())
        scores.append(step_score)         # computed on CPU
        loss.backward()
        optimizer.step()

        # show information
        if (batch_idx + 1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accu: {:.2f}%'.format(
                epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item(), 100 * step_score))
    print(losses, scores)
    return losses, scores


def validation(model, device, optimizer, test_loader):
    # set model as testing mode
    cnn_encoder, rnn_decoder=model
    cnn_encoder.eval()
    rnn_decoder.eval()
    test_loss = 0
    all_y = []
    all_y_pred = []
    with torch.no_grad():
        for X, y in test_loader:
            # distribute data to device
            X, y = X.to(device), y.to(device).view(-1, )

            output = rnn_decoder(cnn_encoder(X))

            loss = F.cross_entropy(output, y, reduction='sum')
            test_loss += loss.item()                 # sum up batch loss
            y_pred = output.max(1, keepdim=True)[1]  # (y_pred != output) get the index of the max log-probability
            print(y,y_pred)
            # collect all y and y_pred in all batches
            all_y.extend(y)
            all_y_pred.extend(y_pred)
            

    test_loss /= len(test_loader.dataset)

    # to compute accuracy
    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)
    test_score = accuracy_score(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())

    # show information
    print('\nTest set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(len(all_y), test_loss, 100* test_score))

    print(save_model_path)
    # save Pytorch models of best record
    torch.save(cnn_encoder.state_dict(), os.path.join(save_model_path, 'cnn_encoder_epoch{}.pth'.format(epoch + 1)))  # save spatial_encoder
    torch.save(rnn_decoder.state_dict(), os.path.join(save_model_path, 'rnn_decoder_epoch{}.pth'.format(epoch + 1)))      # save optimizer
    torch.save(optimizer.state_dict(),os.path.join(save_model_path, 'optimizer_epoch{}.pth'.format(epoch+1)))
    print("Epoch {} model saved!".format(epoch + 1))

    return test_loss, test_score




########################
#Detect devices
use_cuda=torch.cuda.is_available()
device=torch.device('cuda' if use_cuda else'cpu')

#prepare data
le=LabelEncoder()
ansfile=open('answers.txt','r')
num=[0 for i in range(trynum)]
gestures=[0 for i in range(trynum)] #y
filearray=[]    #load data to this array
lis=ansfile.readlines()
for videonum in range(trynum):
    num[videonum], gestures[videonum]=lis[videonum].split()
    filearray.append('outfile{:02d}'.format(videonum))
    #action_name_path='names/'
    #save_model_path='saves/'
names=['shaking','nodding']
le.fit(names)
action_category = le.transform(gestures).reshape(-1,1)
enc = OneHotEncoder()
enc.fit(action_category)
#yarray=enc.transform(actions).toarray()
yarray=labels2cat(le,gestures)
print(yarray)
train_list, test_list, train_label, test_label = train_test_split(filearray, yarray, test_size=test_size, random_state=42)

#############################train!

# image transformation  
transform = transforms.Compose([transforms.Resize([res_size,res_size]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225])])

selected_frames = np.arange(begin_frame, end_frame, skip_frame).tolist()

train_set, valid_set = Dataset_CRNN(data_path, train_list, train_label, selected_frames, transform=transform), \
                       Dataset_CRNN(data_path, test_list, test_label, selected_frames, transform=transform)
print('valid set:',valid_set)
params={'batch_size': batch_size, 'shuffle':True, 'num_workers':0, 'pin_memory':True}# if use_cuda else{}
train_loader = data.DataLoader(train_set, **params)
valid_loader = data.DataLoader(valid_set, **params)

#create model
cnn_encoder=ResCNNEncoder(fc_hidden1=CNN_fc_hidden1,fc_hidden2=CNN_fc_hidden2,drop_p=dropout_p, CNN_embed_dim=CNN_embed_dim).to(device) 
rnn_decoder=DecoderRNN(CNN_embed_dim=CNN_embed_dim, h_RNN_layers=RNN_hidden_layers, h_RNN=RNN_hidden_nodes,h_FC_dim=RNN_FC_dim, drop_p=dropout_p, num_classes=k).to(device)

cnn_encoder=nn.DataParallel(cnn_encoder)
rnn_decoder=nn.DataParallel(rnn_decoder)
#Parallelize model to multiple GPUs
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    #combine encoderCNN+decoderRNN
    crnn_params = list(cnn_encoder.module.fc1.parameters()) + list(cnn_encoder.module.bn1.parameters())+list(cnn_encoder.module.fc2.parameters()) + list(cnn_encoder.module.bn2.parameters())+list(cnn_encoder.module.fc3.parameters()) + list(rnn_decoder.parameters())

elif torch.cuda.device_count()==1:
    print("Using", torch.cuda.device_count(), "GPU!")
    crnn_params = list(cnn_encoder.module.fc1.parameters()) + list(cnn_encoder.module.bn1.parameters()) +list(cnn_encoder.module.fc2.parameters()) + list(cnn_encoder.module.bn2.parameters()) +list(cnn_encoder.module.fc3.parameters()) + list(rnn_decoder.parameters())
else:
    print("Not GPU!")
    crnn_params = list(cnn_encoder.module.fc1.parameters()) + list(cnn_encoder.module.bn1.parameters()) +list(cnn_encoder.module.fc2.parameters()) + list(cnn_encoder.module.bn2.parameters()) +list(cnn_encoder.module.fc3.parameters()) + list(rnn_decoder.parameters())

optimizer=torch.optim.Adam(crnn_params, lr=learning_rate)

optimizer = torch.optim.Adam(crnn_params, lr=learning_rate)   # optimize all cnn parameters

# record training process
epoch_train_losses = []
epoch_train_scores = []
epoch_test_losses = []
epoch_test_scores = []

# start training
for epoch in range(epochs):
    # train, test model
    train_losses, train_scores = train(log_interval, [cnn_encoder,rnn_decoder], device, train_loader, optimizer, epoch)
    epoch_test_loss, epoch_test_score = validation([cnn_encoder,rnn_decoder], device, optimizer, valid_loader)

    # save results
    epoch_train_losses.append(train_losses)
    epoch_train_scores.append(train_scores)
    epoch_test_losses.append(epoch_test_loss)
    epoch_test_scores.append(epoch_test_score)

    # save all train test results
    A = np.array(epoch_train_losses)
    B = np.array(epoch_train_scores)
    C = np.array(epoch_test_losses)
    D = np.array(epoch_test_scores)
    np.save('./CRNN_epoch_training_losses.npy', A)
    np.save('./CRNN_epoch_training_scores.npy', B)
    np.save('./CRNN_epoch_test_loss.npy', C)
    np.save('./CRNN_epoch_test_score.npy', D)





'''

# load UCF101 actions names
with open(action_name_path, 'rb') as f:
    action_names = pickle.load(f)   # load UCF101 actions names

# convert category -> 1-hot
action_category = le.transform(action_names).reshape(-1, 1)
enc = OneHotEncoder()
enc.fit(action_category)

actions = []
fnames = os.listdir(data_path)

all_names = []
for f in fnames:
    loc1 = f.find('v_')
    loc2 = f.find('_g')
    actions.append(f[(loc1 + 2): loc2])

    all_names.append(f)


# list all data files
all_X_list = all_names              # all video file names
all_y_list = labels2cat(le, actions)    # all video labels

'''
# plot
fig = plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.plot(np.arange(1, epochs + 1), A[:, -1])  # train loss (on epoch end)
plt.plot(np.arange(1, epochs + 1), C)         #  test loss (on epoch end)
plt.title("model loss")
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['train', 'test'], loc="upper left")
# 2nd figure
plt.subplot(122)
plt.plot(np.arange(1, epochs + 1), B[:, -1])  # train accuracy (on epoch end)
plt.plot(np.arange(1, epochs + 1), D)         #  test accuracy (on epoch end)
# plt.plot(histories.losses_val)
plt.title("training scores")
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['train', 'test'], loc="upper left")
title = "./fig_RwsNetCRNN.png"
plt.savefig(title, dpi=600)
# plt.close(fig)
plt.show()
