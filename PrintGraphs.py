#---------- Import ----------#
import pandas as pd
import matplotlib.pyplot as plt

#---------------------------------------------------------------- FEDADP ----------------------------------------------------------------------------------------#

#---------- IID----------#

#---------- Import DataBase ----------#
dataAvgBN = pd.read_csv("Results\FedADP_results\FedAVG_5_local_ep_Norm_Batch Norm_iid_1_lr_0.01_mom_0.5_epochs_50.csv")
dataAvgGN = pd.read_csv("Results\FedADP_results\FedAVG_5_local_ep_Norm_Group Norm_iid_1_lr_0.01_mom_0.5_epochs_50.csv")
dataMix025075 = pd.read_csv("Results\FedADP_results\FedMix_5_local_ep_iid_1_lr_0.01_mom_0.5_epochs_50_alphaB_0.25_alphaG_0.75.csv")
dataMix0109 = pd.read_csv("Results\FedADP_results\FedMix_5_local_ep_iid_1_lr_0.01_mom_0.5_epochs_50_alphaB_0.1_alphaG_0.9.csv")
dataMix0010 = pd.read_csv("Results\FedADP_results\FedMix_5_local_ep_iid_1_lr_0.01_mom_0.5_epochs_50_alphaB_0.0_alphaG_1.0.csv")
dataMix0505 = pd.read_csv("Results\FedADP_results\FedMix_5_local_ep_iid_1_lr_0.01_mom_0.5_epochs_50_alphaB_0.5_alphaG_0.5.csv")

#---------- Plot the graphs ----------#
plt.figure(figsize=(10,10))
plt.plot(dataAvgBN["Epochs"],dataAvgBN["Test accuracy"],marker = "o",markersize=3.5,linestyle = "--", label = 'FedAvg IID Batch Norm')
plt.plot(dataAvgGN["Epochs"],dataAvgGN["Test accuracy"],marker = "o",markersize=3.5,linestyle = "--", label = 'FedAvg IID Group Norm')
plt.plot(dataMix025075["Epochs"],dataMix025075["Test accuracy"],marker = "o",markersize=3.5, label = 'FedMix IID alpha_b=0.25 alpha_g=0.75')
plt.plot(dataMix0109["Epochs"],dataMix0109["Test accuracy"],marker = "o",markersize=3.5, label = 'FedMix IID alpha_b=0.1 alpha_g=0.9')
plt.plot(dataMix0010["Epochs"],dataMix0010["Test accuracy"],marker = "o",markersize=3.5, label = 'FedMix IID alpha_b=0.0 alpha_g=1.0')
plt.plot(dataMix0505["Epochs"],dataMix0505["Test accuracy"],marker = "o",markersize=3.5, label = 'FedMix IID alpha_b=0.5 alpha_g=0.5')
plt.ylim(bottom = 0)
plt.title("Comparison between methods with IID in FedAdp settings")
plt.xlabel("Epochs")
plt.ylabel("Test Accuracy")
plt.grid()
plt.legend(loc="lower right")
plt.show()

#---------- NONIID----------#

#---------- Import DataBase ----------#
dataAvgBN = pd.read_csv("Results\FedADP_results\FedAVG_5_local_ep_Norm_Batch Norm_iid_0_lr_0.01_mom_0.5_epochs_50.csv")
dataAvgGN = pd.read_csv("Results\FedADP_results\FedAVG_5_local_ep_Norm_Group Norm_iid_0_lr_0.01_mom_0.5_epochs_50.csv")
dataMix025075 = pd.read_csv("Results\FedADP_results\FedMix_5_local_ep_iid_0_lr_0.01_mom_0.5_epochs_50_alphaB_0.25_alphaG_0.75.csv")
dataMix0109 = pd.read_csv("Results\FedADP_results\FedMix_5_local_ep_iid_0_lr_0.01_mom_0.5_epochs_50_alphaB_0.1_alphaG_0.9.csv")
dataMix0010 = pd.read_csv("Results\FedADP_results\FedMix_5_local_ep_iid_0_lr_0.01_mom_0.5_epochs_50_alphaB_0.0_alphaG_1.0.csv")
dataMix0505 = pd.read_csv("Results\FedADP_results\FedMix_5_local_ep_iid_0_lr_0.01_mom_0.5_epochs_50_alphaB_0.5_alphaG_0.5.csv")

#---------- Plot the graphs ----------#
plt.figure(figsize=(10,10))
plt.plot(dataAvgBN["Epochs"],dataAvgBN["Test accuracy"],marker = "o",markersize=3.5,linestyle = "--", label = 'FedAvg NonIID Batch Norm')
plt.plot(dataAvgGN["Epochs"],dataAvgGN["Test accuracy"],marker = "o",markersize=3.5,linestyle = "--", label = 'FedAvg NonIID Group Norm')
plt.plot(dataMix025075["Epochs"],dataMix025075["Test accuracy"],marker = "o",markersize=3.5, label = 'FedMix IID alpha_b=0.25 alpha_g=0.75')
plt.plot(dataMix0109["Epochs"],dataMix0109["Test accuracy"],marker = "o",markersize=3.5, label = 'FedMix NonIID alpha_b=0.1 alpha_g=0.9')
plt.plot(dataMix0010["Epochs"],dataMix0010["Test accuracy"],marker = "o",markersize=3.5, label = 'FedMix NonIID alpha_b=0.0 alpha_g=1.0')
plt.plot(dataMix0505["Epochs"],dataMix0505["Test accuracy"],marker = "o",markersize=3.5, label = 'FedMix IID alpha_b=0.5 alpha_g=0.5')
plt.ylim(bottom = 0)
plt.title("Comparison between methods with NonIID in FedAdp settings")
plt.xlabel("Epochs")
plt.ylabel("Test Accuracy")
plt.grid()
plt.legend(loc="lower right")
plt.show()

#---------------------------------------------------------------- FEDMIX ----------------------------------------------------------------------------------------#


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
plt.title("Comparison between methods with NonIID in FedMix settings ")
plt.xlabel("Epochs")
plt.ylabel("Test Accuracy")
plt.grid()
plt.legend(loc="lower right")
plt.show()


#---------- IID----------#

#---------- Import DataBase ----------#
dataAvgBN = pd.read_csv("Results\FedAVG_results\FedAVG_5_local_ep_Norm_Batch Norm_iid_1_lr_0.01_mom_0.5_epochs_50.csv")
dataAvgGN = pd.read_csv("Results\FedAVG_results\FedAVG_5_local_ep_Norm_Group Norm_iid_1_lr_0.01_mom_0.5_epochs_50.csv")
dataMix0505 = pd.read_csv("Results\FedMIX_results\FedMix_5_local_ep_iid_1_lr_0.01_mom_0.5_epochs_50_alphaB_0.5_alphaG_0.5.csv")
dataMix0109 = pd.read_csv("Results\FedMIX_results\FedMix_5_local_ep_iid_1_lr_0.01_mom_0.5_epochs_50_alphaB_0.1_alphaG_0.9.csv")
dataMix0901 = pd.read_csv("Results\FedMIX_results\FedMix_5_local_ep_iid_1_lr_0.01_mom_0.5_epochs_50_alphaB_0.9_alphaG_0.1.csv")

#---------- Plot the graphs ----------#
plt.figure(figsize=(10,10))
plt.plot(dataAvgBN["Epochs"],dataAvgBN["Test accuracy"],marker = "o",markersize=3.5,linestyle = "--", label = 'FedAvg IID Batch Norm')
plt.plot(dataAvgGN["Epochs"],dataAvgGN["Test accuracy"],marker = "o",markersize=3.5,linestyle = "--", label = 'FedAvg IID Group Norm')
plt.plot(dataMix0505["Epochs"],dataMix0505["Test accuracy"],marker = "o",markersize=3.5, label = 'FedMix IID alpha_b=0.5 alpha_g=0.5')
plt.plot(dataMix0109["Epochs"],dataMix0109["Test accuracy"],marker = "o",markersize=3.5, label = 'FedMix IID alpha_b=0.1 alpha_g=0.9')
plt.plot(dataMix0901["Epochs"],dataMix0901["Test accuracy"],marker = "o",markersize=3.5, label = 'FedMix IID alpha_b=0.9 alpha_g=0.1')
plt.ylim(bottom = 0)
plt.title("Comparison between methods with IID in FedMix setting")
plt.xlabel("Epochs")
plt.ylabel("Test Accuracy")
plt.grid()
plt.legend(loc="lower right")
plt.show()


#---------------------------------------------------------------- FEDGKT 128 local bs----------------------------------------------------------------------------------------#


#---------- NONIID----------#
#---------- Import DataBase ----------#
#dataAvgBN = pd.read_csv("Results\FedAVG_results\FedAVG_5_local_ep_Norm_Batch Norm_iid_1_lr_0.01_mom_0.5_epochs_50.csv")
#dataAvgGN = pd.read_csv("Results\FedAVG_results\FedAVG_5_local_ep_Norm_Group Norm_iid_1_lr_0.01_mom_0.5_epochs_50.csv")
#dataMix0505 = pd.read_csv("Results\FedMIX_results\FedMix_5_local_ep_iid_1_lr_0.01_mom_0.5_epochs_50_alphaB_0.5_alphaG_0.5.csv")
#dataMix0109 = pd.read_csv("Results\FedMIX_results\FedMix_5_local_ep_iid_1_lr_0.01_mom_0.5_epochs_50_alphaB_0.1_alphaG_0.9.csv")
#dataMix0901 = pd.read_csv("Results\FedMIX_results\FedMix_5_local_ep_iid_1_lr_0.01_mom_0.5_epochs_50_alphaB_0.9_alphaG_0.1.csv")
##---------- Plot the graphs ----------#
#plt.figure(figsize=(10,10))
#plt.plot(dataAvgBN["Epochs"],dataAvgBN["Test accuracy"],marker = "o",markersize=3.5,linestyle = "--", label = 'FedAvg IID Batch Norm')
#plt.plot(dataAvgGN["Epochs"],dataAvgGN["Test accuracy"],marker = "o",markersize=3.5,linestyle = "--", label = 'FedAvg IID Group Norm')
#plt.plot(dataMix0505["Epochs"],dataMix0505["Test accuracy"],marker = "o",markersize=3.5, label = 'FedMix IID alpha_b=0.5 alpha_g=0.5')
#plt.plot(dataMix0109["Epochs"],dataMix0109["Test accuracy"],marker = "o",markersize=3.5, label = 'FedMix IID alpha_b=0.1 alpha_g=0.9')
#plt.plot(dataMix0901["Epochs"],dataMix0901["Test accuracy"],marker = "o",markersize=3.5, label = 'FedMix IID alpha_b=0.9 alpha_g=0.1')
#plt.ylim(bottom = 0)
#plt.title("Comparison between methods with NonIID in FedGKT setting")
#plt.xlabel("Communication rounds")
#plt.ylabel("Test Accuracy")
#plt.grid()
#plt.legend(loc="lower right")
#plt.show()

#---------- IID----------#
#---------- Import DataBase ----------#
dataAvgBN = pd.read_csv("Results\FedGKT_results\_128 local_bs\FedGKT_Batch_Norm_IID.csv")
dataAvgGN = pd.read_csv("Results\FedGKT_results\_128 local_bs\FedGKT_Group_Norm_IID.csv")
dataMix0505 = pd.read_csv("Results\FedGKT_results\_128 local_bs\Fedgkt_test_results_iid_1_unbalance_0_alphaB_0.5_alphaG_0.5.csv")
dataMix0109 = pd.read_csv("Results\FedGKT_results\_128 local_bs\Fedgkt_test_results_iid_1_unbalance_0_alphaB_0.1_alphaG_0.9.csv")
dataMix0901 = pd.read_csv("Results\FedGKT_results\_128 local_bs\Fedgkt_test_results_iid_1_unbalance_0_alphaB_0.9_alphaG_0.1.csv")

#---------- Plot the graphs ----------#
plt.figure(figsize=(10,10))
plt.plot(dataAvgBN["round"],dataAvgBN["test_acc"],marker = "o",markersize=3.5,linestyle = "--", label = 'FedGKT IID Batch Norm')
plt.plot(dataAvgGN["round"],dataAvgGN["test_acc"],marker = "o",markersize=3.5,linestyle = "--", label = 'FedGKT IID Group Norm')
plt.plot(dataMix0505["round"]+1,dataMix0505["test_acc"]/100,marker = "o",markersize=3.5, label = 'FedGKTMix IID alpha_b=0.5 alpha_g=0.5')
plt.plot(dataMix0109["round"]+1,dataMix0109["test_acc"]/100,marker = "o",markersize=3.5, label = 'FedGKTMix IID alpha_b=0.1 alpha_g=0.9')
plt.plot(dataMix0901["round"]+1,dataMix0901["test_acc"]/100,marker = "o",markersize=3.5, label = 'FedGKTMix IID alpha_b=0.9 alpha_g=0.1')
plt.ylim(bottom = 0)
plt.title("Comparison between methods with IID in FedGKT setting 128 l_bs")
plt.xlabel("Communication rounds")
plt.ylabel("Test Accuracy")
plt.grid()
plt.legend(loc="lower right")
plt.show()





#---------------------------------------------------------------- FEDGKT 16 local bs----------------------------------------------------------------------------------------#

#---------- IID----------#
#---------- Import DataBase ----------#
dataAvgBN = pd.read_csv("Results\FedGKT_results\_16 local_bs\Fedgkt_test_results_Norm_Batch Norm_iid_1_unbalance_0.csv")
dataAvgGN = pd.read_csv("Results\FedGKT_results\_16 local_bs\Fedgkt_test_results_Norm_Group Norm_iid_1_unbalance_0.csv")
dataMix0505 = pd.read_csv("Results\FedGKT_results\_16 local_bs\Fedgkt_test_results_iid_1_unbalance_0_alphaB_0.5_alphaG_0.5.csv")
dataMix0109 = pd.read_csv("Results\FedGKT_results\_16 local_bs\Fedgkt_test_results_iid_1_unbalance_0_alphaB_0.1_alphaG_0.9.csv")
dataMix0901 = pd.read_csv("Results\FedGKT_results\_16 local_bs\Fedgkt_test_results_iid_1_unbalance_0_alphaB_0.9_alphaG_0.1.csv")

#---------- Plot the graphs ----------#
plt.figure(figsize=(10,10))
plt.plot(dataAvgBN["round"],dataAvgBN["test_acc"]/100,marker = "o",markersize=3.5,linestyle = "--", label = 'FedGKT IID Batch Norm')
plt.plot(dataAvgGN["round"],dataAvgGN["test_acc"]/100,marker = "o",markersize=3.5,linestyle = "--", label = 'FedGKT IID Group Norm')
plt.plot(dataMix0505["round"],dataMix0505["test_acc"]/100,marker = "o",markersize=3.5, label = 'FedGKTMix IID alpha_b=0.5 alpha_g=0.5')
plt.plot(dataMix0109["round"],dataMix0109["test_acc"]/100,marker = "o",markersize=3.5, label = 'FedGKTMix IID alpha_b=0.1 alpha_g=0.9')
plt.plot(dataMix0901["round"],dataMix0901["test_acc"]/100,marker = "o",markersize=3.5, label = 'FedGKTMix IID alpha_b=0.9 alpha_g=0.1')
plt.ylim(bottom = 0)
plt.title("Comparison between methods with IID in FedGKT setting 16 l_bs")
plt.xlabel("Communication rounds")
plt.ylabel("Test Accuracy")
plt.grid()
plt.legend(loc="lower right")
plt.show()











#---------------------------------------------------------------- FEDMIX ----------------------------------------------------------------------------------------#


#---------- NONIID Unbalaced----------#

#---------- Import DataBase ----------#
dataAvgBN = pd.read_csv(r'Results\FedMix-Avg_unbalanced_results\FedAVG_5_local_ep_Norm_Batch Norm_iid_0_lr_0.01_mom_0.5_epochs_50.csv')
dataAvgGN = pd.read_csv(r'Results\FedMix-Avg_unbalanced_results\FedAVG_5_local_ep_Norm_Group Norm_iid_0_lr_0.01_mom_0.5_epochs_50.csv')
dataMix0505 = pd.read_csv(r"Results\FedMix-Avg_unbalanced_results\FedMix_5_local_ep_iid 0_lr 0.01_mom 0.5_epochs 50_alphaB 0.5_alphaG 0.5.csv")
dataMix0109 = pd.read_csv(r"Results\FedMix-Avg_unbalanced_results\FedMix_5_local_ep_iid 0_lr 0.01_mom 0.5_epochs 50_alphaB 0.1_alphaG 0.9.csv")
dataMix0901 = pd.read_csv(r"Results\FedMix-Avg_unbalanced_results\FedMix_5_local_ep_iid 0_lr 0.01_mom 0.5_epochs 50_alphaB 0.9_alphaG 0.1.csv")

#---------- Plot the graphs ----------#
plt.figure(figsize=(10,10))
plt.plot(dataAvgBN["Epochs"],dataAvgBN["Test accuracy"],marker = "o",markersize=3.5,linestyle = "--", label = 'FedAvg NonIID Unbalanced Batch Norm')
plt.plot(dataAvgGN["Epochs"],dataAvgGN["Test accuracy"],marker = "o",markersize=3.5,linestyle = "--", label = 'FedAvg NonIID Unbalanced Group Norm')
plt.plot(dataMix0505["Epochs"],dataMix0505["Test accuracy"],marker = "o",markersize=3.5, label = 'FedMix NonIID Unbalanced alpha_b=0.5 alpha_g=0.5')
plt.plot(dataMix0109["Epochs"],dataMix0109["Test accuracy"],marker = "o",markersize=3.5, label = 'FedMix NonIID Unbalanced alpha_b=0.1 alpha_g=0.9')
plt.plot(dataMix0901["Epochs"],dataMix0901["Test accuracy"],marker = "o",markersize=3.5, label = 'FedMix NonIID Unbalanced alpha_b=0.9 alpha_g=0.1')
plt.ylim(bottom = 0)
plt.title("Comparison between methods with NonIID Unbalanced in FedMix settings ")
plt.xlabel("Epochs")
plt.ylabel("Test Accuracy")
plt.grid()
plt.legend(loc="lower right")
plt.show()


#---------------------------------------------------------------- FEDGKT 128 local bs----------------------------------------------------------------------------------------#


#---------- NONIID Unbalaced----------#
#---------- Import DataBase ----------#
dataAvgBN = pd.read_csv("Results\FedGKT_results\_128 local_bs\FedGKT_Batch_Norm_Non_IID_Unbalanced.csv")
dataAvgGN = pd.read_csv("Results\FedGKT_results\_128 local_bs\FedGKT_Group_Norm_Non_IID_Unbalanced.csv")
dataMix0505 = pd.read_csv("Results\FedGKT_results\_128 local_bs\Fedgkt_test_results_iid_0_unbalance_1_alphaB_0.5_alphaG_0.5.csv")
dataMix0109 = pd.read_csv("Results\FedGKT_results\_128 local_bs\Fedgkt_test_results_iid_0_unbalance_1_alphaB_0.1_alphaG_0.9.csv")
dataMix0901 = pd.read_csv("Results\FedGKT_results\_128 local_bs\Fedgkt_test_results_iid_0_unbalance_1_alphaB_0.9_alphaG_0.1.csv")

#---------- Plot the graphs ----------#
plt.figure(figsize=(10,10))
plt.plot(dataAvgBN["round"],dataAvgBN["test_acc"],marker = "o",markersize=3.5,linestyle = "--", label = 'FedGKT NonIID Unbalanced Batch Norm')
plt.plot(dataAvgGN["round"],dataAvgGN["test_acc"],marker = "o",markersize=3.5,linestyle = "--", label = 'FedGKT NonIID Unbalanced Group Norm')
plt.plot(dataMix0505["round"]+1,dataMix0505["test_acc"]/100,marker = "o",markersize=3.5, label = 'FedGKTMix NonIID Unbalanced alpha_b=0.5 alpha_g=0.5')
plt.plot(dataMix0109["round"]+1,dataMix0109["test_acc"]/100,marker = "o",markersize=3.5, label = 'FedGKTMix NonIID Unbalanced alpha_b=0.1 alpha_g=0.9')
plt.plot(dataMix0901["round"]+1,dataMix0901["test_acc"]/100,marker = "o",markersize=3.5, label = 'FedGKTMix NonIID Unbalanced alpha_b=0.9 alpha_g=0.1')
plt.ylim(bottom = 0)
plt.title("Comparison between methods with NonIID Unbalanced in FedGKT setting 128 l_bs")
plt.xlabel("Communication rounds")
plt.ylabel("Test Accuracy")
plt.grid()
plt.legend(loc="lower right")
plt.show()



#---------------------------------------------------------------- FEDGKT 16 local bs----------------------------------------------------------------------------------------#


#---------- NONIID Unbalaced----------#
#---------- Import DataBase ----------#
dataAvgBN = pd.read_csv("Results\FedGKT_results\_16 local_bs\Fedgkt_test_results_Norm_Batch Norm_iid_0_unbalance_1.csv")
dataAvgGN = pd.read_csv("Results\FedGKT_results\_16 local_bs\Fedgkt_test_results_Norm_Group Norm_iid_0_unbalance_1.csv")
dataMix0505 = pd.read_csv("Results\FedGKT_results\_16 local_bs\Fedgkt_test_results_iid_0_unbalance_1_alphaB_0.5_alphaG_0.5.csv")
dataMix0109 = pd.read_csv("Results\FedGKT_results\_16 local_bs\Fedgkt_test_results_iid_0_unbalance_1_alphaB_0.1_alphaG_0.9.csv")
dataMix0901 = pd.read_csv("Results\FedGKT_results\_16 local_bs\Fedgkt_test_results_iid_0_unbalance_1_alphaB_0.9_alphaG_0.1.csv")

#---------- Plot the graphs ----------#
plt.figure(figsize=(10,10))
plt.plot(dataAvgBN["round"],dataAvgBN["test_acc"]/100,marker = "o",markersize=3.5,linestyle = "--", label = 'FedGKT NonIID Unbalanced Batch Norm')
plt.plot(dataAvgGN["round"],dataAvgGN["test_acc"]/100,marker = "o",markersize=3.5,linestyle = "--", label = 'FedGKT NonIID Unbalanced Group Norm')
plt.plot(dataMix0505["round"],dataMix0505["test_acc"]/100,marker = "o",markersize=3.5, label = 'FedGKTMix NonIID Unbalanced alpha_b=0.5 alpha_g=0.5')
plt.plot(dataMix0109["round"],dataMix0109["test_acc"]/100,marker = "o",markersize=3.5, label = 'FedGKTMix NonIID Unbalanced alpha_b=0.1 alpha_g=0.9')
plt.plot(dataMix0901["round"],dataMix0901["test_acc"]/100,marker = "o",markersize=3.5, label = 'FedGKTMix NonIID Unbalanced alpha_b=0.9 alpha_g=0.1')
plt.ylim(bottom = 0)
plt.title("Comparison between methods with NonIID Unbalanced in FedGKT setting 16 l_bs")
plt.xlabel("Communication rounds")
plt.ylabel("Test Accuracy")
plt.grid()
plt.legend(loc="lower right")
plt.show()