# Federated Learning And Federated Mix

## Contributions 

This repo was built by [Luca Marcellino](https://github.com/LucaMarcellino), [Luca Villani](https://github.com/lucavillanigit) and [Edoardo Bonelli](https://github.com/wh33li3). We worked all together to guarantee the best possible results.

## Content

In this repository, you will find a different implementation of the Federated Learning methods. First, you can find an implementation related to FederatedAVG and another related to FedGKT.

Finally, you can find two possible new implementations, to the best of our knowledge, that we called **Federated Mix** and **Federated Adaptive**.

## Federated Mix

This implementation is based on FedAvg but with an important difference. Indeed the latter method implements Batch Norm or Group Norm. In our method, we tried to merge both normalizations using linear combinations.

## Federated Adp

In this method, we want to generalize the idea of batch size. Indeed we decided a local batch size for each client and applied FedAvg to have a look at the results. 



