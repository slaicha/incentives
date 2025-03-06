"""adapted from pytorch-example: A Flower / PyTorch app."""

import torch
import json
import numpy as np
from src.task import Net, get_weights, load_data, set_weights, test, train, pretrain

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, ParametersRecord, RecordSet, array_from_numpy

seed = 4
np.random.seed(4)
torch.manual_seed(4)  # PyTorch CPU
torch.cuda.manual_seed(4)  # PyTorch GPU
torch.cuda.manual_seed_all(4)  # Multi-GPU


class FlowerClient(NumPyClient):
    """A simple client that showcases how to use the state.

    It implements a basic version of `personalization` by which
    the classification layer of the CNN is stored locally and used
    and updated during `fit()` and used during `evaluate()`.
    """

    def __init__(
        self, net, client_state: RecordSet, trainloader, valloader, local_epochs
    ):
        self.net: Net = net
        self.client_state = client_state
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.local_layer_name = "classification-head"
        self.initial_lr = 0.1
        self.decay_rate = 0.996

    def fit(self, parameters, config):
        """Train model locally.

        The client stores in its context the parameters of the last layer in the model
        (i.e. the classification head). The classifier is saved at the end of the
        training and used the next time this client participates.
        """

        # I'm not sure if I should be saving the classification head during round 1

        # Apply weights from global models (the whole model is replaced)
        set_weights(self.net, parameters)
        
        current_round = config["server_round"]
        
        if current_round < 3:
            lr = self.initial_lr 
        else:
            lr = self.initial_lr * (self.decay_rate ** (current_round-2))  
        # print(f"Server Round: {current_round}, Learning Rate: {lr}")
        
        train_loss = train(
            self.net,
            self.trainloader,
            self.local_epochs,
            lr=lr,
            # lr=float(config["lr"]),
            device=self.device,
        )

        return (
            get_weights(self.net),
            len(self.trainloader.dataset),
            {"train_loss": train_loss},
        )
        

    def evaluate(self, parameters, config):
        """Evaluate the global model on the local validation set.

        Note the classification head is replaced with the weights this client had the
        last time it trained the model.
        """
        set_weights(self.net, parameters)
        # Override weights in classification layer with those this client
        # had at the end of the last fit() round it participated in
        # self._load_layer_weights_from_state()
        loss, accuracy = test(self.net, self.valloader, self.device)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}

    
def client_fn(context: Context):
    # Load model and data
    net = Net()
    net_pretrain = Net()  # This model is only used for pretraining to estimate Gn.
    node_id = context.node_id
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    client_state = context.state
    trainloader, valloader = load_data(partition_id, num_partitions)
    local_epochs = context.run_config["local-epochs"]
    pretrain_local_epochs = context.run_config["pretrain-local-epochs"]
        
    # Check if pretraining has already been done for this client. All pretraining will happen in round 1
    if "pretrained" not in client_state.parameters_records:
        # print(f"Pretraining client {partition_id}...")
        Gn_squared, _ = pretrain(
            net_pretrain, trainloader, epochs=pretrain_local_epochs, lr=0.01, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        
        config_path = "/home/as1233/incentives/incentives/config_utility.json"  
        with open(config_path, "r") as f:
            config = json.load(f)
        
        config["N"] = num_partitions

        client_state.parameters_records["pretrained"] = ParametersRecord({
            "Gn_squared": array_from_numpy(np.array(Gn_squared))
        })
        
        if "G_n_squared" not in config:
            config["G_n_squared"] = {}
            
        config["G_n_squared"][node_id] = Gn_squared # add the new values

        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)  # Save with indentation for readability

    # Return FlowerClient instance for federated learning
    return FlowerClient(
        net, client_state, trainloader, valloader, local_epochs
    ).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
