#---------- Import ----------#
import pandas as pd
import matplotlib.pyplot as plt


#---------- NONIID----------#

#---------- Import DataBase ----------#
dataAvgBN = pd.read_csv("Results\FedAVG_results\FedAVG_5_local_ep_Norm_Batch Norm_iid_0_lr_0.01_mom_0.5_epochs_50.csv")
dataAvgGN = pd.read_csv("Results\FedAVG_results\FedAVG_5_local_ep_Norm_Group Norm_iid_0_lr_0.01_mom_0.5_epochs_50.csv")
dataMix0505 = pd.read_csv("Results\FedMIX_results\FedMix_5_local_ep_iid_0_lr_0.01_mom_0.5_epochs_50_alphaB_0.5_alphaG_0.5.csv")
dataMix0109 = pd.read_csv("Results\FedMIX_results\FedMix_5_local_ep_iid_0_lr_0.01_mom_0.5_epochs_50_alphaB_0.1_alphaG_0.9.csv")
dataMix0901 = pd.read_csv("Results\FedMIX_results\FedMix_5_local_ep_iid_0_lr_0.01_mom_0.5_epochs_50_alphaB_0.9_alphaG_0.1.csv")

#---------- Plot the graphs ----------#
plt.figure(figsize=(10,10))
plt.plot(dataAvgBN["Epochs"],dataAvgBN["Test accuracy"],marker = "o",markersize=3.5,linestyle = "--", label = 'FedAvg NonIID Batch Norm')
plt.plot(dataAvgGN["Epochs"],dataAvgGN["Test accuracy"],marker = "o",markersize=3.5,linestyle = "--", label = 'FedAvg NonIID Group Norm')
plt.plot(dataMix0505["Epochs"],dataMix0505["Test accuracy"],marker = "o",markersize=3.5, label = 'FedMix NonIID alpha_b=0.5 alpha_g=0.5')
plt.plot(dataMix0109["Epochs"],dataMix0109["Test accuracy"],marker = "o",markersize=3.5, label = 'FedMix NonIID alpha_b=0.1 alpha_g=0.9')
plt.plot(dataMix0901["Epochs"],dataMix0901["Test accuracy"],marker = "o",markersize=3.5, label = 'FedMix NonIID alpha_b=0.9 alpha_g=0.1')
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
dataAvgGN = pd.read_csv("Results\FedAVG_results\FedAVG_5_local_ep_Norm_Group Norm_iid_1_lr_0.01_mom_0.5_epochs_50.csv")


#---------- Plot the graphs ----------#
plt.figure(figsize=(10,10))
plt.plot(dataAvgBN["Epochs"],dataAvgBN["Test accuracy"],marker = "o",markersize=3.5,linestyle = "--", label = 'FedAvg IID Batch Norm')
plt.plot(dataAvgGN["Epochs"],dataAvgGN["Test accuracy"],marker = "o",markersize=3.5,linestyle = "--", label = 'FedAvg IID Group Norm')
plt.ylim(bottom = 0)
plt.title("Comparison between methods With IID")
plt.xlabel("Epochs")
plt.ylabel("Test Accuracy")
plt.grid()
plt.legend(loc="lower right")
plt.show()
