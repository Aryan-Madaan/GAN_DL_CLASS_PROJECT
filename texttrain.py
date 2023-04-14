# %%
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import os
device = "cuda"
# %%
trainx = np.load("trainX.npy")
trainy = np.load("trainY.npy")

# %%
trainy[1].min()

# %%
trainX = torch.tensor(trainx[:trainy.shape[0]]).to(device)
trainY = torch.tensor(trainy).to(device)

# %%
trainX.dtype

# %%

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.hidden1 = nn.Linear(89, 1024)
        self.hidden2= nn.Linear(1024, 1024)
        self.output = nn.Linear(1024, 512)
        
        # Define sigmoid activation and softmax output 
        self.relu =nn.LeakyReLU()
        #self.sigmoid = nn.Tanh()
        
    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hidden1(x)
        x = self.relu(x)
        x = self.hidden2(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.relu(x)
        #x = torch.mul(x,torch.tensor(7.0))
        
        return x

# %%
model = Network()
if os.path.exists("text_encoder/mymodel"):
    model = torch.load("./text_encoder/mymodel")
    print("loaded model from ./text_encoder/mymodel")
model = model.double().to(device)

# %%
batch_size = 10
num_epochs = 10000
num_batches_per_train_epoch = trainX.shape[0] // batch_size

# %%
optimizer = optim.Adam(model.parameters(), lr = 0.001)
criterion = nn.MSELoss()

# %%
# Train the model

epoch_loss = 0
losses_at_each_epoch = list()
train_accuracies = list()

# Forward pass -> Backward pass -> Weight update

for epoch in range(num_epochs):
  epoch_loss = 0

  model.train()  

  # Train one epoch
  for batch_idx in range(num_batches_per_train_epoch):
    optimizer.zero_grad()  # This line is necessary to flush out the gradients of the previous batch. 

    input = trainX[batch_idx*batch_size: (batch_idx+1)*batch_size] # Slice out batch_size amount of the training data
    output = model(input)  
    target_out = trainY[batch_idx*batch_size: (batch_idx+1)*batch_size]

    batch_loss = criterion(output, target_out)

    batch_loss.backward()
    optimizer.step()   # We take a single gradient descent step on the parameters of our model here
                       # By this we mean => parameter_new = parameter_old - lr*grad(loss,parameters)

    epoch_loss += batch_loss
  losses_at_each_epoch.append(epoch_loss.detach() / batch_size)


  if epoch % 1000 == 0:
    print("Epoch %2i " % (
                epoch))
    print(losses_at_each_epoch[-1])


# %%
my_ans = model(torch.tensor(trainx[1]).to(device)).cpu().detach().numpy()

torch.save(model,"text_encoder/mymodel")


