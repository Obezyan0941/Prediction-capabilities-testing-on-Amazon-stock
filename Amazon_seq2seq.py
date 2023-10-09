import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.linalg import vector_norm
from torch.nn.functional import normalize

data = pd.read_csv('AMZN.csv')
data = data[['Date', 'Close']]

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

data['Date'] = pd.to_datetime(data['Date'])


def data_prep(dataframe, num_steps):
    num_steps -= 1
    df = dataframe
    # df.set_index('Date', inplace=True)
    for i in range(1, num_steps + 1):
        df[f'Close(t-{i})'] = df['Close'].shift(i)

    df.dropna(inplace=True)
    df = df[df.columns[::-1]]
    return df


steps = 28
prdict_len = 8
dataset = data_prep(data, steps)
x_df = dataset.drop('Date', axis=1)
y_df = dataset['Date']

data_tensor = torch.tensor(x_df.values)
data_norm = vector_norm(data_tensor, dim=None)
data_normalized = normalize(data_tensor, dim=None)

split_index = int(data_normalized.size()[0] * 0.95)

x_tensor, y_tensor = data_normalized[:, :(steps - prdict_len)], data_normalized[:, (steps - prdict_len):]
x_train, x_test = x_tensor[:split_index, :], x_tensor[split_index:, :]
y_train, y_test = y_tensor[:split_index, :], y_tensor[split_index:, :]

x_train = x_train.view(-1, (steps - prdict_len), 1).float()
y_train = y_train.view(-1, prdict_len, 1).float()
x_test = x_test.view(-1, (steps - prdict_len), 1).float()
y_test = y_test.view(-1, prdict_len, 1).float()


# print("\n", x_train.size(), "\n", y_train.size(), "\n", x_test.size(), "\n", y_test.size())


class TimeSeriesDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


train_dataset = TimeSeriesDataset(x_train, y_train)
test_dataset = TimeSeriesDataset(x_test, y_test)

batch_size = 8

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers, batch_first=True)

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)
        return output, hidden, cell


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        prediction = self.fc(output)

        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(input_size, hidden_size, num_layers)
        self.decoder = Decoder(input_size, hidden_size, num_layers, output_size)

    def forward(self, x):
        encoder_output, encoder_hidden, encoder_cell = self.encoder(x)
        h, c = encoder_hidden, encoder_cell
        decoder_out, _, _ = self.decoder(x[:, -1, :].unsqueeze(1), h, c)
        decoder_out = decoder_out.permute(0, 2, 1)

        return decoder_out


def count_trend(pred, test, current_batch_num):
    total, valid = 0, 0
    current_sample = []
    current_trend = []
    current_is_true = []
    consolidation_percent = 0.03
    for num, i in enumerate(pred):
        check_pred = 0
        check_test = 0
        first_num_test = test[num][0].item()
        last_num_test = test[num][-1].item()
        if abs((first_num_test - last_num_test)) / first_num_test < consolidation_percent:
            check_test = 1
        else:
            if first_num_test < last_num_test:
                check_test = 2
            elif first_num_test > last_num_test:
                check_test = 3
        first_num_pred = pred[num][0].item()
        last_num_pred = pred[num][-1].item()
        if abs((first_num_pred - last_num_pred)) / first_num_pred < consolidation_percent:
            check_pred = 1
        else:
            if first_num_pred < last_num_pred:
                check_pred = 2
            elif first_num_pred > last_num_pred:
                check_pred = 3
        total += 1
        current_sample.append("Batch %d, sample %d" % (current_batch_num, total))
        if check_pred == check_test:
            valid += 1
            current_is_true.append("True")
        else:
            current_is_true.append("False")
        if check_test == 1:
            current_trend.append("Consolidation")
        if check_test == 2:
            current_trend.append("Uptrend")
        if check_test == 3:
            current_trend.append("Downtrend")

    current_trend_data_df = pd.DataFrame({"Sample №": current_sample, "Trend": current_trend, "Prediction": current_is_true})
    return total, valid, current_trend_data_df


def training(isload=True, issave=True, dict_nom=""):
    model = Seq2Seq(1, prdict_len, 1, prdict_len)
    if isload:
        model.load_state_dict(torch.load(dict_nom))
        print("State has been loaded.")
    model.to(device)
    model.train(True)
    learning_rate = 0.001
    num_epochs = 200
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, num_epochs):
        print(f'Epoch: {epoch}')
        running_loss = 0.0
        batches_trained = 0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_pred = model(x_batch)
            loss = loss_function(y_pred.squeeze(), y_batch.squeeze())
            running_loss += loss
            batches_trained += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = running_loss / batches_trained
        print('Loss: %f' % avg_loss)

    if issave:
        torch.save(model.state_dict(), dict_nom)
        print("Parameters saved.")

    print("Training finished.")


def testing(dict_nom="", isplot=True, to_excel=True):
    model = Seq2Seq(1, prdict_len, 1, prdict_len)
    model.to(device)
    model.eval()
    model.load_state_dict(torch.load(dict_nom))
    loss_function = nn.MSELoss()

    final_y_pred = []
    final_y_test = []
    total_batches = 0
    total_valid = 0
    batch_count = 0
    trend_data_df = pd.DataFrame({"Sample №": ['None'], "Trend": ['None'], "Prediction": ['None']}, index=[0])

    for x_batch, y_batch in test_loader:
        batch_count += 1
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        y_pred = model(x_batch)
        loss = loss_function(y_pred.squeeze(), y_batch.squeeze())
        print(loss)

        for i in y_pred:
            i = i * data_norm
            final_y_pred.append(i[0].item())
        for l in y_batch:
            l = l * data_norm
            final_y_test.append(l[0].item())

        y_pred = y_pred * data_norm
        y_batch = y_batch * data_norm
        total_, valid_, temp_df = count_trend(y_pred, y_batch, batch_count)
        trend_data_df = pd.concat([trend_data_df, temp_df], ignore_index=True)
        total_batches += total_
        total_valid += valid_
        if isplot:
            y_pred = y_pred[0].squeeze(0)
            y_batch = y_batch[0].squeeze(0)
            y_pred = y_pred.flatten().cpu().detach().numpy()
            y_batch = y_batch.flatten().cpu().detach().numpy()
            plt.plot(y_batch, label="Actual Close")
            plt.plot(y_pred, label="Predicted Close")
            plt.xlabel('Day')
            plt.ylabel('Close')
            plt.legend()
            plt.show()

    if to_excel:
        trend_data_df.to_excel('count_trend_seq2seq.xlsx', index=False)
    print("\nTotal batches: ", total_batches, "\nValid trends: ", total_valid)
    print("Trend prediction accuracy: ", round(total_valid/total_batches*100, 2), "%")


# training(isload=True, issave=True, dict_nom="Seq2Seq_Amazon_3.pth")
testing(dict_nom="Seq2Seq_Amazon_3.pth", isplot=False)
