#---------- Import ----------#
import pandas as pd
import matplotlib.pyplot as plt


#---------- NONIID----------#

#---------- Import DataBase ----------#
dataAvgBN = pd.read_csv("Results\FedAVG_results\FedAVG_5_local_ep_Norm_Batch Norm_iid_0_lr_0.01_mom_0.5_epochs_50.csv")

#---------- Plot the graphs ----------#
plt.figure(figsize=(10,10))
plt.plot(dataAvgBN["Epochs"],dataAvgBN["Test accuracy"],marker = "o", label = 'FedAvg NonIID Batch Norm')
plt.ylim(bottom = 0)
plt.title("Comparison between methods With NonIID")
plt.xlabel("Epochs")
plt.ylabel("Test Accuracy")
plt.grid()
plt.legend(loc="lower right")
plt.show()
