import torch
import numpy as np
import random
import pandas as pd
from tqdm import tqdm

from gkt_models_server import ResNet50
from gkt_models_client import ResNet8
from gkt_server import GKTServerTrainer
from gkt_client import GKTClientTrainer

from options import args_parser
from utils import exp_details,get_dataset,average_weights

def seed_worker(args):
	worker_seed = args.seed
	np.random.seed(worker_seed)
	random.seed(worker_seed)


if __name__ == "__main__":

	#Retrieve Arguments
	args = args_parser()
	exp_details(args)

	#Dataset Import
	train_dataset, test_dataset, user_groups = get_dataset(args)

	seed_worker(args)


	#Choosing norm
	if args.num_groups == 0:
		norm_type = "BatchNorm"
	elif args.num_groups > 0:
		norm_type = "GroupNorm"

	#Init Models
	server_model = ResNet50(n_type = norm_type)
	client_model = ResNet8(n_type = norm_type)

	#Device choice
	device = torch.device("cpu") if args.gpu == 0 else torch.device("cuda")
	server_model.to(device)
	client_model.to(device)

	server_model.train()
	client_model.train()

	server_trainer = GKTServerTrainer(server_model,device,args)

	#Client Init
	clients_trainer = []
	idxs_users = range(args.num_users)
	
	for idx in idxs_users:
		#self, model, device,  train_dataset, test_dataset, idxs, client_index, args
		client_trainer = GKTClientTrainer(client_model,device,train_dataset,test_dataset
										,user_groups[idx], idx, args)
		clients_trainer.append(client_trainer)

	#Training
	for communication_round in tqdm(range(1, args.communication_rounds+1)):
	    print(f'\nCommunication Round: {communication_round} \n')

	    max_chosen_users = max(int(args.frac * args.num_users), 1) # number of users to be used for federated updates, at least 1
	    idxs_chosen_users = np.random.choice(range(args.num_users), max_chosen_users, replace=False) # choose randomly m users

	    print("Chosen users indexes: ",idxs_chosen_users)
	    for idx in idxs_chosen_users:
	        # the server broadcast k-th Z_c to the client
	        extracted_feature_dict, logits_dict, labels_dict, extracted_feature_dict_test,\
	        labels_dict_test = clients_trainer[idx].train()
	        # send client result to server
	        server_trainer.add_local_trained_result(idx, extracted_feature_dict, logits_dict, labels_dict,
	                                                extracted_feature_dict_test, labels_dict_test)

	    server_trainer.train(communication_round, idxs_chosen_users)

	    for idx in idxs_chosen_users:
	        # get global logits
	        global_logits = server_trainer.get_global_logits(idx)

	        # send global logits to client
	        clients_trainer[idx].update_large_model_logits(global_logits)
	
	# get lists of train loss and accuracy
	train_loss, train_accuracy = server_trainer.get_loss_acc_list()

	file_name = 'data/seed:{}_{}_{}_{}_lr_[{}]_C[{}]_iid[{}]_Es[{}]_Ec[{}]_B[{}]_{}_unbalanced[{}].csv'.\
    format(args.seed,"ResNet50", norm_type, args.communication_rounds, args.lr, args.frac, args.iid,
           args.epochs, args.local_ep, args.local_bs, args.optimizer, args.unequal)

	data = list(zip(train_loss, train_accuracy))
	result = pd.DataFrame(data, columns=['Communication_round','train_loss','train_accuracy'])
	result.to_csv(file_name)



		
