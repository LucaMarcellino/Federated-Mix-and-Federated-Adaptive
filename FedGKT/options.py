import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs_server', type=int, default=10,
                        help="number of rounds of training")
    parser.add_argument('--epochs_client', type=int, default=1,
                        help="number of rounds of training")                    
    parser.add_argument('--num_users', type=int, default=100,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1,
                        help='the fraction of clients: C')
    parser.add_argument('--local_bs', type=int, default=16,
                        help="local batch size: B")
    parser.add_argument('--communication_rounds', type=int, default=50)

    # model arguments
    parser.add_argument('--norm_layer', type=str, default='Batch Norm',
                        help="Batch Norm, Group Norm")


    # other arguments
    parser.add_argument('--gpu', default=None, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    args = parser.parse_args()
    return args