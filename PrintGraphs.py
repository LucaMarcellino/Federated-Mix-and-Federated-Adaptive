#---------- Import ----------#
import pandas as pd
import matplotlib.pyplot as plt


#---------- NONIID----------#

#---------- Import DataBase ----------#
dataAvgBN = pd.read_csv("Results\FedAVG_results\FedAVG_5_local_ep_Norm_Batch Norm_iid_0_lr_0.01_mom_0.5_epochs_50.csv")
dataMix0505 = pd.read_csv("Results\FedMIX_results\FedMix_5_local_ep_iid_0_lr_0.01_mom_0.5_epochs_50_alphaB_0.5_alphaG_0.5.csv")

#---------- Plot the graphs ----------#
plt.figure(figsize=(10,10))
plt.plot(dataAvgBN["Epochs"],dataAvgBN["Test accuracy"],marker = "o", label = 'FedAvg NonIID Batch Norm')
plt.plot(dataMix0505["Epochs"],dataMix0505["Test accuracy"],marker = "o", label = 'FedMix NonIID alpha_b=0.5 alpha_g=0.5')
plt.ylim(bottom = 0)
plt.title("Comparison between methods With NonIID")
plt.xlabel("Epochs")
plt.ylabel("Test Accuracy")
plt.grid()
plt.legend(loc="lower right")
plt.show()


#---------- IID----------#

#---------- Import DataBase ----------#
dataAvgBN = pd.read_csv("Results\FedAVG_results\FedAVG_5_local_ep_Norm_Batch Norm_iid_1_lr_0.01_mom_0.5_epochs_50.csv")


#---------- Plot the graphs ----------#
plt.figure(figsize=(10,10))
plt.plot(dataAvgBN["Epochs"],dataAvgBN["Test accuracy"],marker = "o", label = 'FedAvg IID Batch Norm')
plt.ylim(bottom = 0)
plt.title("Comparison between methods With IID")
plt.xlabel("Epochs")
plt.ylabel("Test Accuracy")
plt.grid()
plt.legend(loc="lower right")
plt.show()
