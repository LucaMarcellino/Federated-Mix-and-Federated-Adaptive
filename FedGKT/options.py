import argparse

def args_parser():

    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=10,
                        help="number of rounds of training")#
    parser.add_argument('--num_users', type=int, default=100,
                        help="number of users: K")#
    parser.add_argument('--frac', type=float, default=0.1,
                        help='the fraction of clients: C')#
    parser.add_argument('--local_ep', type=int, default=1,
                        help="the number of local epochs: E")#
    parser.add_argument('--local_bs', type=int, default=128,
                        help="local batch size: B")#
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')#
    parser.add_argument('--loss', type=str, default = "CrossEntropyLoss",
                        help='loss function used by the models')#
    parser.add_argument('--partition_alpha',type=float,default=0.5,
                        help='alpha partition of the clients')#
    parser.add_argument('--temperature',type=float, default = 3.0,
                        help='Parameter for Kloss')#[TODO] check 


    # other arguments
    parser.add_argument('--num_groups', type=int, default=0, help="number \
                        of groups - (0 for BatchNorm) - (> 0 for GroupNorm")
    parser.add_argument('--gpu', default=1, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")#
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer (sgd or adam)")#
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')#
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')#
    parser.add_argument('--communication_rounds', type=int, default=10,
                        help='rounds of early stopping')#
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    args = parser.parse_args()
    return args
