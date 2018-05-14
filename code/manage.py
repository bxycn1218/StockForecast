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

choice = 1
if choice == 0:
    res1, acc1, mae1 = lstm.train_lstm(data, input_size, output_size, lr, time_step, choice=0)
    res2, acc2, mae2 = bp.train_bp(data, input_size, output_size, lr, choice=0)
    # plot acc and mae
    plt.figure()
    plt.title('acc')
    plt.xlabel('Number of iterations')
    plt.ylabel('value')
    plt.plot(list(range(len(acc1))), acc1, color='green')
    plt.plot(list(range(len(acc2))), acc2, color='red')
    plt.show()

    plt.figure()
    plt.title('mae')
    plt.xlabel('Number of iterations')
    plt.ylabel('value')
    plt.plot(list(range(len(mae1))), mae1, color='green')
    plt.plot(list(range(len(mae2))), mae2, color='red')
    plt.show()
else:
    lstm.train_lstm(data, input_size, output_size, lr, time_step, choice=1)
    predict_lstm, test_y, acc_lstm, mae_lstm = lstm.prediction(data, input_size, output_size)
    bp.train_bp(data, input_size, output_size, lr, choice=1)
    predict_bp, test_y, acc_bp, mae_bp = bp.prediction(data, input_size, output_size)

    # plot
    plt.figure()
    plt.title('stock prediction')
    plt.xlabel('time')
    plt.ylabel('value')
    plt.plot(list(range(len(predict_bp))), predict_bp, color='red', label='bp')
    plt.plot(list(range(len(test_y))), test_y, color='black', label='original')
    plt.plot(list(range(len(predict_lstm))), predict_lstm, color='green', label='lstm')
    plt.legend(loc=2)
    plt.show()

    # print lstm and bp
    print("lstm acc:", acc_lstm, " lstm mae:", mae_lstm)
    print("bp acc:", acc_bp, " bp mae:", mae_bp)

