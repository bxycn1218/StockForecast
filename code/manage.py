import pandas as pd
import matplotlib.pyplot as plt
import lstm
import bp


# import data
f = open('dataset/dataset_1.csv')
# read stock info
df = pd.read_csv(f)
# get col 3-10
data = df.iloc[:, 2:10].values

# set parameter
input_size = 7   # input size
output_size = 1  # output size
lr = 0.001       # learning rate
time_step = 20   # time step for lstm

# lstm
lstm.train_lstm(data, input_size, output_size, lr, time_step)
predict_lstm, test_y, acc_lstm, mae_lstm = lstm.prediction(data, input_size, output_size)

# bp
bp.train_bp(data, input_size, output_size, lr)
predict_bp, test_y, acc_bp, mae_bp = bp.prediction(data, input_size, output_size)

# plot
plt.figure()
plt.plot(list(range(len(predict_bp))), predict_bp, color='red')
plt.plot(list(range(len(test_y))), test_y, color='black')
plt.plot(list(range(len(predict_lstm))), predict_lstm, color='green')
plt.show()

print("lstm acc:", acc_lstm, " lstm mae:", mae_lstm)
print("bp acc:", acc_bp, " bp mae:", mae_bp)