import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import time

class lamParam_strain(torch.utils.data.Dataset):
    def __init__(self, train, size):
        # Specify GPU training which is supposed to be faster than CPU training
        # I did not see a time difference, but this is a simple neural net
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        dataset = 'reduced' # or 'expanded'

        # Data import and cleanup
        if dataset == 'reduced':
            data = pd.read_csv(r'C:\Users\carne\Desktop\AE 8900 Code & Data\Machine Learning\training_reduced.csv')
            data = data.drop(['e1', 'e2', 'e3', 'e12', 'e13', 'e23'], axis=1)
            data = data.rename({'eI': 'e1', 'eII': 'e2', 'eIII': 'e3'}, axis=1)
        elif dataset == 'expanded':
            data = pd.read_csv(r'C:\Users\carne\Desktop\AE 8900 Code & Data\Machine Learning\training_expanded.csv')
            data = data.rename({'eI': 'e1', 'eII': 'e2', 'eIII': 'e3'}, axis=1)

        # Gets either training data (first 'size' of .csv) or testing data (last 'size' of .csv)
        if train:
            data = data.loc[data['ID'] < size]
        else:
            data = data.loc[data['ID'] >= len(data['ID']) - size]

        # The input & output variables to the NN
        inputs = ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'D1', 'D2', 'D3']
        output = ['e1', 'e2', 'e3']

        # Getting the data from pandas format to PyTorch tensor
        self.X = torch.tensor(data[inputs].values, dtype=torch.float32).to(device)
        self.y = torch.tensor(data[output].values, dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        # Three sub-neural nets, each corresponding to one principal strain output
        self.e1nn = nn.Sequential(
            nn.Linear(9, 128),
            nn.ReLU(),
            nn.Linear(128, 1))
        self.e2nn = nn.Sequential(
            nn.Linear(9, 128),
            nn.ReLU(),
            nn.Linear(128, 1))
        self.e3nn = nn.Sequential(
            nn.Linear(9, 128),
            nn.ReLU(),
            nn.Linear(128, 1))

    def freeze(self, strain):
        if strain == 'e1':
            set_grad(self.e1nn, True)
            set_grad(self.e2nn, False)
            set_grad(self.e3nn, False)            
        elif strain == 'e2':
            set_grad(self.e1nn, False)
            set_grad(self.e2nn, True)
            set_grad(self.e3nn, False)
        elif strain == 'e3':
            set_grad(self.e1nn, False)
            set_grad(self.e2nn, False)
            set_grad(self.e3nn, True)

    def forward(self, x):
        e1 = self.e1nn(x)
        e2 = self.e2nn(x)
        e3 = self.e3nn(x)
        return torch.cat([e1, e2, e3], 1)

def pgnnLoss(outputs, targets, strain):
    e1 = targets[:, 0]
    e2 = targets[:, 1]
    e3 = targets[:, 2]
    e1nn = outputs[:, 0]
    e2nn = outputs[:, 1]
    e3nn = outputs[:, 2]

    # Physics guided portion
    # loss = abs(e1nn - e1) + abs(e2nn - e2) + abs(e3nn - e3) \
    #      + abs(e1nn**3 - (e1 + e2 + e3)*e1nn**2 + (e1*e2 + e1*e3 + e2*e3)*e1nn - e1*e2*e3) \
    #      + abs(e2nn**3 - (e1 + e2 + e3)*e2nn**2 + (e1*e2 + e1*e3 + e2*e3)*e2nn - e1*e2*e3) \
    #      + abs(e3nn**3 - (e1 + e2 + e3)*e3nn**2 + (e1*e2 + e1*e3 + e2*e3)*e3nn - e1*e2*e3)

    # if strain == 'e1':
    #     loss = abs(e1nn - e1) + abs(e1nn**3 - (e1nn + e2 + e3)*e1nn**2 + (e1nn*e2 + e1nn*e3 + e2*e3)*e1nn - e1nn*e2*e3)
    # elif strain == 'e2':
    #     loss = abs(e2nn - e2) + abs(e2nn**3 - (e1 + e2nn + e3)*e2nn**2 + (e1*e2nn + e1*e3 + e2nn*e3)*e2nn - e1*e2nn*e3)
    # elif strain == 'e3':
    #     loss = abs(e3nn - e3) + abs(e3nn**3 - (e1 + e2 + e3nn)*e3nn**2 + (e1*e2 + e1*e3nn + e2*e3nn)*e3nn - e1*e2*e3nn)

    # if strain == 'e1':
    #     loss = abs(e1nn - e1) + abs(e1nn**3 - (e1 + e2 + e3)*e1nn**2 + (e1*e2 + e1*e3 + e2*e3)*e1nn - e1nn*e2*e3)
    # elif strain == 'e2':
    #     loss = abs(e2nn - e2) + abs(e2nn**3 - (e1 + e2 + e3)*e2nn**2 + (e1*e2 + e1*e3 + e2*e3)*e2nn - e1*e2nn*e3)
    # elif strain == 'e3':
    #     loss = abs(e3nn - e3) + abs(e3nn**3 - (e1 + e2 + e3)*e3nn**2 + (e1*e2 + e1*e3 + e2*e3)*e3nn - e1*e2*e3nn)

    if strain == 'e1':
        loss = abs(e1nn**3 - (e1 + e2 + e3)*e1nn**2 + (e1*e2 + e1*e3 + e2*e3)*e1nn - e1nn*e2*e3)
    elif strain == 'e2':
        loss = abs(e2nn**3 - (e1 + e2 + e3)*e2nn**2 + (e1*e2 + e1*e3 + e2*e3)*e2nn - e1*e2nn*e3)
    elif strain == 'e3':
        loss = abs(e3nn**3 - (e1 + e2 + e3)*e3nn**2 + (e1*e2 + e1*e3 + e2*e3)*e3nn - e1*e2*e3nn)

    # Averaging error
    # loss = sum(loss)/len(loss)

    # Mean-Squaring error
    loss = sum(loss**2)/len(loss)

    return loss

def set_grad(model, TF):
    for param in model.parameters():
        param.requires_grad = TF

if __name__ == '__main__':
    
    # Showing CPU or GPU processing
    print(f'Using: {torch.cuda.get_device_name(0)}\n')

    # Reproducibility
    torch.manual_seed(0)

    # Training & testing data setup
    batchSize = 16
    trainSize = 580
    testSize = 20

    # Setting the training & testing data
    trainingData = lamParam_strain(train=True, size=trainSize)
    testingData = lamParam_strain(train=False, size=testSize)

    trainLoader = torch.utils.data.DataLoader(trainingData, batch_size=batchSize, shuffle=True, num_workers=0)

    # Starting the NN settings
    epochs = 800
    epochDisplay = epochs/50
    learningRate = 1e-4

    mlp = MLP().to('cuda')
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=learningRate)

    displayData = []
    for epoch in range(epochs):
        if epoch%epochDisplay == 0:
            startTime = time.time()

        epochLoss = 0
        for i, data in enumerate(trainLoader):
            # Get and prepare inputs
            inputs, targets = data

            strains = ['e1', 'e2', 'e3']
            for strain in strains:
                mlp.freeze(strain)

                # Zero the gradients
                optimizer.zero_grad()

                # Perform forward pass
                outputs = mlp(inputs)

                # loss = pgnnLoss(outputs, targets, strain)
                if strain == 'e1':
                    loss = loss_function(outputs[:, 0], targets[:, 0])
                elif strain == 'e2':
                    loss = loss_function(outputs[:, 1], targets[:, 1])
                elif strain == 'e3':
                    loss = loss_function(outputs[:, 2], targets[:, 2])
                epochLoss += loss

                # Perform backward pass
                loss.backward()

                # Perform optimization
                optimizer.step()

        # Display & log progress
        if (epoch + 1)%epochDisplay == 0:
            testExact = pd.DataFrame(data=testingData.y.cpu().numpy(), columns=['e1', 'e2', 'e3'])
            testNN = pd.DataFrame(data=mlp(testingData.X).cpu().detach().numpy(),  columns=['e1nn', 'e2nn', 'e3nn'])
            testResults = testExact.join(testNN)
            testResults['e1err'] = (testResults['e1nn'] - testResults['e1']).abs()/testResults['e1'].abs()*100
            testResults['e2err'] = (testResults['e2nn'] - testResults['e2']).abs()/testResults['e2'].abs()*100
            testResults['e3err'] = (testResults['e3nn'] - testResults['e3']).abs()/testResults['e3'].abs()*100

            displayData.append([epochLoss.cpu().data.item(), 
                                testResults['e1err'].mean(),
                                testResults['e2err'].mean(),
                                testResults['e3err'].mean()])
            
            print(f'Finished Epoch {epoch + 1}')
            print(f'e1 Avg Pct Error: {testResults["e1err"].mean():.2f}%')
            print(f'e2 Avg Pct Error: {testResults["e2err"].mean():.2f}%')
            print(f'e3 Avg Pct Error: {testResults["e3err"].mean():.2f}%')
            print(f'Epoch Loss: {epochLoss:.2f}')
            print(f'Time taken: {time.time()-startTime:.2f} (s)\n')

    # Process is complete.
    print('Training process has finished.')
    torch.save(mlp.state_dict(), 'model.pth')

    # Plot losses & error
    displayData = pd.DataFrame(data=displayData, columns=['losses', 'e1', 'e2', 'e3'])
    displayData.plot(y=['losses', 'e1', 'e2', 'e3'], subplots=True, layout=(4, 1), kind='line', sharex='true')
    plt.show()
