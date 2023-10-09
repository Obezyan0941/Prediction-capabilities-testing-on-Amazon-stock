import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math
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


steps = 56
prdict_len = 26
dataset = data_prep(data, steps)
x_df = dataset.drop('Date', axis=1)
y_df = dataset['Date']

data_tensor = torch.tensor(x_df.values)
# data_norm = vector_norm(data_tensor, dim=None)
# data_normalized = normalize(data_tensor, dim=None)
data_normalized = data_tensor

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


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)

        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1)  # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(
            torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model)  # 1000^(2i/dim_model)

        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)

        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])


class Transformer(nn.Module):

    # Constructor
    def __init__(
            self,
            dim_model,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            dropout_p,
    ):
        super().__init__()

        # INFO
        self.model_type = "Transformer"
        self.dim_model = dim_model

        # LAYERS
        self.input_embedding = nn.Linear(1, dim_model)
        self.positional_encoder = PositionalEncoding(
            dim_model=dim_model, dropout_p=dropout_p, max_len=5000
        )
        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout_p,
            batch_first=True,
            norm_first=True
        )
        self.out = nn.Linear(dim_model, 1)

    def forward(self, src, tgt, tgt_mask=None):
        # print("tgt out ", tgt.size())
        # print("src", src.size())
        # print("tgt", tgt.size())
        src = self.input_embedding(src)
        tgt = self.input_embedding(tgt)
        # print("src", src.size())
        # print("tgt", tgt.size())

        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)
        # print("positional_encoder tgt out ", tgt.size())
        # print("src", src.size())
        # print("tgt", tgt.size())

        transformer_out = self.transformer(src, tgt, tgt_mask=tgt_mask)
        # print("tr out ", transformer_out.size())
        out = self.out(transformer_out)
        # print("lin out ", out.size())

        return out

    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1)  # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf'))  # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0))  # Convert ones to 0

        return mask


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


def training(transformer_model, isload=True, issave=True, num_epochs=200, dict_nom=""):
    if isload:
        transformer_model.load_state_dict(torch.load(dict_nom))
        print("State has been loaded.")
    transformer_model.to(device)
    transformer_model.train(True)
    learning_rate = 0.0001
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(transformer_model.parameters(), lr=learning_rate)

    for epoch in range(1, num_epochs+1):
        print(f'Epoch: {epoch}')
        running_loss = 0.0
        batches_trained = 0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            whole_batch = torch.cat([x_batch, y_batch], dim=1)
            data_norm = vector_norm(whole_batch, dim=None)
            whole_batch = normalize(whole_batch, dim=None)
            x_batch = whole_batch[:, :steps-prdict_len]
            y_batch = whole_batch[:, steps-prdict_len:]

            y_input = y_batch[:, :-1]
            y_expected = y_batch[:, 1:]
            tgt_mask = transformer_model.get_tgt_mask(y_input.size(1)).to(device)
            y_pred = transformer_model(x_batch, y_input, tgt_mask)

            # print(x_batch.size())
            # print(y_input.size())
            # print(y_expected.size())
            # print(y_pred[0])
            # quit()

            loss = loss_function(y_pred, y_expected)
            batches_trained += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.detach().item()

        avg_loss = running_loss / batches_trained
        print('Loss: ', avg_loss)

        if issave:
            torch.save(transformer_model.state_dict(), dict_nom)
            print("Parameters saved.")

    print("Training completed.")


def testing(transformer_model, dict_nom="", isplot=True, to_excel=True):
    transformer_model.eval()
    transformer_model.load_state_dict(torch.load(dict_nom))
    loss_function = nn.MSELoss()

    batch_count = 0
    total_batches = 0
    total_valid = 0
    running_loss = 0.0
    trend_data_df = pd.DataFrame({"Sample №": ['None'], "Trend": ['None'], "Prediction": ['None']}, index=[0])

    for x_batch, y_batch in test_loader:
        batch_count += 1
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        whole_batch = torch.cat([x_batch, y_batch], dim=1)
        data_norm = vector_norm(whole_batch, dim=None)
        whole_batch = normalize(whole_batch, dim=None)
        x_batch = whole_batch[:, :steps-prdict_len]
        y_batch = whole_batch[:, steps-prdict_len:]

        window_size = steps-prdict_len-1
        x_batch_src = x_batch[:, :-1]
        x_batch_tgt = x_batch[:, -1:]
        # print(x_batch_tgt[0])
        # print(x_batch_tgt[:, -window_size:, :].size())

        pred_seq = torch.rand(y_batch.size()).to(device)
        pred_seq[:, 0, :] = x_batch_tgt.squeeze(1)
        # print("pred_seq[0]: \n", pred_seq[0])
        # print("pred_seq[0].size(): \n", pred_seq.size())
        # quit()

        # for i in range(0, y_batch.size(1)):
        #     tgt_mask = model.get_tgt_mask(pred_seq.size(1)).to(device)
        #     with torch.no_grad():
        #         pred = transformer_model(x_batch_src, pred_seq, tgt_mask)
        #     pred_seq[:, i, :] = pred.squeeze(1)
        #     # pred_seq = torch.cat([pred_seq, pred[:, -1:]], dim=1)
        #     print("pred_seq[0]: \n", pred_seq[0])
        #     input()

        tgt_mask = model.get_tgt_mask(pred_seq.size(1)).to(device)
        with torch.no_grad():
            pred = transformer_model(x_batch_src, pred_seq, tgt_mask)

        x_batch_mean = x_batch.mean(dim=1).unsqueeze(1)
        pred_mean = pred.mean(dim=1).unsqueeze(1)
        pred = (pred / pred_mean) * x_batch_mean
        pred_seq = pred
        loss = loss_function(pred_seq, y_batch)
        running_loss += loss.detach().item()

        total_, valid_, temp_df = count_trend(pred_seq, y_batch, batch_count)
        trend_data_df = pd.concat([trend_data_df, temp_df], ignore_index=True)
        total_batches += total_
        total_valid += valid_
        if isplot:
            y_pred = pred_seq[:, 1:][0].squeeze(0)*data_norm
            y_batch = y_batch[:, 1:][0].squeeze(0)*data_norm
            y_pred = y_pred.flatten().cpu().detach().numpy()
            y_batch = y_batch.flatten().cpu().detach().numpy()
            plt.plot(y_batch, label="Actual Close")
            plt.plot(y_pred, label="Predicted Close")
            plt.xlabel('Day')
            plt.ylabel('Close')
            plt.legend()
            plt.show()
    avg_loss = running_loss / total_batches
    if to_excel:
        trend_data_df.to_excel('count_trend_transformer.xlsx', index=False)
    print('Loss: ', avg_loss)
    print("\nTotal batches: ", total_batches, "\nValid trends: ", total_valid)
    print("Trend prediction accuracy: ", round(total_valid / total_batches * 100, 2), "%")


def adjustment_mode(transformer_model, isload=False, dict_nom=''):
    if isload:
        transformer_model.load_state_dict(torch.load(dict_nom))
        print("State has been loaded.")
    for x_batch, y_batch in test_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        whole_batch = torch.cat([x_batch, y_batch], dim=1)
        data_norm = vector_norm(whole_batch, dim=None)
        whole_batch = normalize(whole_batch, dim=None)
        x_batch = whole_batch[:, :x_batch.size(1)]
        y_batch = whole_batch[:, y_batch.size(1):]

        print("x_batch: ", x_batch.size())
        print("y_batch: ", y_batch.size())
        y_input = y_batch[:, :-1]
        y_expected = y_batch[:, 1:]
        print("y_input: ", y_input.size())
        print("y_expected: ", y_expected.size())
        tgt_mask = transformer_model.get_tgt_mask(y_input.size(1)).to(device)
        print("tgt_mask: ", tgt_mask.size())
        y_pred = transformer_model(x_batch, y_input, tgt_mask)
        print("y_pred: ", y_pred.size())
        print(y_input[0])
        print(y_pred.unsqueeze(2)[0])
        break


model = Transformer(dim_model=300, num_heads=30, num_encoder_layers=20, num_decoder_layers=20,
                    dropout_p=0.1).to(device)
model_parameters_dict = "Transformer_Amazon_9.pth"
# training(model, isload=False, issave=True, num_epochs=20, dict_nom=model_parameters_dict)
testing(model, dict_nom=model_parameters_dict, isplot=True)
# adjustment_mode(model, isload=False, dict_nom=model_parameters_dict)

# dict_nom 2 - dim_model=20, num_heads=10, num_encoder_layers=3, num_decoder_layers=3, dropout_p=0.1; norm_first = False

# dict_nom 3 -dim_model=20, num_heads=10, num_encoder_layers=5, num_decoder_layers=5,
#                     dropout_p=0.1; norm_first = True; embedding_layer = True; steps = 32
# dict_nom 4 -dim_model=320, num_heads=10, num_encoder_layers=5, num_decoder_layers=5,
#                     dropout_p=0.1; norm_first = True; embedding_layer = True ]; steps = 76
# dict_nom 5 -dim_model=300, num_heads=10, num_encoder_layers=5, num_decoder_layers=5,
#                     dropout_p=0.1; norm_first = True; embedding_layer = True ]; steps = 56, normalized;
# dict_nom 6 -dim_model=300, num_heads=10, num_encoder_layers=5, num_decoder_layers=5,
#                     dropout_p=0.1; norm_first = True; embedding_layer = True ]; steps = 56, batch normalized;
# dict_nom 7 -dim_model=300, num_heads=10, num_encoder_layers=5, num_decoder_layers=5,
#                     dropout_p=0.1; norm_first = False; embedding_layer = True ]; steps = 56, batch normalized;
# dict_nom 8 -dim_model=300, num_heads=30, num_encoder_layers=5, num_decoder_layers=5,
#                     dropout_p=0.1; norm_first = True; embedding_layer = True ]; steps = 56, batch normalized;
# dict_nom 9 -dim_model=300, num_heads=30, num_encoder_layers=20, num_decoder_layers=20,
#                     dropout_p=0.1; norm_first = True; embedding_layer = True ]; steps = 56, batch normalized;
