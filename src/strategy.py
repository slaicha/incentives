"""adapted from pytorch-example: A Flower / PyTorch app."""

import json
import random
import numpy as np
from logging import INFO, WARNING
from typing import Optional, Union

import torch
import wandb
from pytorch_example.task import Net, create_run_dir, set_weights

from pytorch_example.utility import solve_optimization

from flwr.common import logger, parameters_to_ndarrays, FitRes, parameters_to_ndarrays
from flwr.common.typing import UserConfig
from flwr.server.strategy import FedAvg
from flwr.server.client_manager import ClientManager as client_manager
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy


from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    FitIns,
    FitRes,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

PROJECT_NAME = "FLOWER-advanced-pytorch"

seed = 4
random.seed(seed) 
np.random.seed(seed)
torch.manual_seed(seed)  # PyTorch CPU
torch.cuda.manual_seed(seed)  # PyTorch GPU
torch.cuda.manual_seed_all(seed)  # Multi-GPU


class SamplingFedAvg(FedAvg):
    """A modified FedAvg strategy implementing arbitrary client sampling probability.

    This strategy:
    (1) Samples clients with probability q_n
    (2) Uses unbiased global model aggregation
    (3) Saves results to the filesystem and logs to W&B if enabled
    """

    def __init__(self, run_config: UserConfig, use_wandb: bool, sampling_ratio, initial_params, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Create a directory where to save results from this run
        self.save_path, self.run_dir = create_run_dir(run_config)
        self.use_wandb = use_wandb
        if use_wandb:
            self._init_wandb_project()

        self.best_acc_so_far = 0.0
        self.results = {} # store results dictionary
        self.latest_model = None
        self.client_probs = None
        self.initial_params = initial_params
        
        self.sampling_ratio = sampling_ratio

    def _init_wandb_project(self):
        wandb.init(project=PROJECT_NAME, name=f"{str(self.run_dir)}-ServerApp")

    def _store_results(self, tag: str, results_dict):
        """Store results and save to JSON."""
        if tag in self.results:
            self.results[tag].append(results_dict)
        else:
            self.results[tag] = [results_dict]
        
        # Save results to disk.
        # Note we overwrite the same file with each call to this function.
        # While this works, a more sophisticated approach is preferred
        # in situations where the contents to be saved are larger.
        with open(f"{self.save_path}/results.json", "w", encoding="utf-8") as fp:
            json.dump(self.results, fp)

    def _update_best_acc(self, round, accuracy, parameters):
        """Check if a new best model has been found and save it."""
        if accuracy > self.best_acc_so_far:
            self.best_acc_so_far = accuracy
            logger.log(INFO, "💡 New best global model found: %f", accuracy)
            # You could save the parameters object directly.
            # Instead we are going to apply them to a PyTorch
            # model and save the state dict.
            # Converts flwr.common.Parameters to ndarrays
            ndarrays = parameters_to_ndarrays(parameters)
            model = Net()
            set_weights(model, ndarrays)
            file_name = f"model_state_acc_{accuracy}_round_{round}.pth"
            torch.save(model.state_dict(), self.save_path / file_name)

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate fit results using a custom weighted aggregation formula."""

        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}
        
        if server_round == 1: # I'm using round 1 for pretraining and not saving the model
            self.latest_model = self.initial_params
            # print("checkpoint 1")
            self.client_probs = self.calculate_client_probs()
            print(self.client_probs)
            return self.latest_model, {}
        
        else: # we started federated training from round 2            
            # Get current global model (w^r) as a NumPy array
            global_model = parameters_to_ndarrays(self.latest_model) # at round 2, latest model is the global initizalized model
            sampling_probabilities = self.client_probs # dict {cid, p_cid}
            
            aggregated_update = [np.zeros_like(layer) for layer in global_model] # Initialize aggregated update

            for client_proxy, fit_res in results:
                client_id = client_proxy.cid
                q_n = sampling_probabilities.get(int(client_id))

                if q_n is None:
                    raise ValueError(f"Sampling probability q_n missing for client {client_id}")

                client_model = parameters_to_ndarrays(fit_res.parameters)
                # Compute model update: (w_n^{r+1} - w^r)
                model_update = [client_layer - global_layer for client_layer, global_layer in zip(client_model, global_model)]
                weight = 1 / (len(self.client_probs) * q_n)
                # Apply weighted model update
                for i in range(len(aggregated_update)):
                    aggregated_update[i] += weight * model_update[i]

            # Compute new global model: w^{r+1} = w^r + aggregated update
            new_global_model = [global_layer + aggregated_layer for global_layer, aggregated_layer in zip(global_model, aggregated_update)]
            self.latest_model = ndarrays_to_parameters(new_global_model) # convert back to flower parameters

            # Aggregate custom metrics if aggregation function was provided
            metrics_aggregated = {}
            if self.fit_metrics_aggregation_fn:
                fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
                metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
            elif server_round == 2:  # Only log this warning once
                log(WARNING, "No fit_metrics_aggregation_fn provided")

            return self.latest_model, metrics_aggregated
    
    def store_results_and_log(self, server_round: int, tag: str, results_dict):
        """Store results and log them to W&B if enabled."""
        self._store_results(
            tag=tag,
            results_dict={"round": server_round, **results_dict},
        )

        if self.use_wandb:
            wandb.log(results_dict, step=server_round)

    def evaluate(self, server_round, parameters):
        """Run centralized evaluation if callback was passed to strategy init."""
        loss, metrics = super().evaluate(server_round, parameters)

        # Save model if new best accuracy is found
        self._update_best_acc(server_round, metrics["centralized_accuracy"], parameters)

        # Store and log
        self.store_results_and_log(
            server_round=server_round,
            tag="centralized_evaluate",
            results_dict={"centralized_loss": loss, **metrics},
        )
        return loss, metrics

    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate results from federated evaluation."""
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)

        # Store and log
        self.store_results_and_log(
            server_round=server_round,
            tag="federated_evaluate",
            results_dict={"federated_evaluate_loss": loss, **metrics},
        )
        return loss, metrics

    def calculate_client_probs(self): # this fct will only be used in the first round to get the sampling prob
        """"Calculate client sampling probabilities"""

        config_path = "/home/as1233/incentives/advanced-pytorch/config_utility.json"  
        with open(config_path, "r") as f:
            config = json.load(f)
        # fix the order of the parameters
        if "Gn" not in config:
            config["Gn"] = []
        if "cid" not in config:
            config["cid"] = []
        # remove sorting G bc i don't think there really is a relationship between G and c
        # sorted_items = sorted(config["G_n_squared"].items(), key=lambda item: item[1], reverse=True)        
        items = config["G_n_squared"].items()      
        config["Gn"] = [item[1] for item in items]
        config["cid"] = [item[0] for item in items]
        
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
        
        q, _ = solve_optimization(scenario=config["scenario"], G=config["Gn"])
                        
        sampling_prob = {}
        for i in range(len(config["cid"])):
            sampling_prob[int(config["cid"][i])] = q[i]

        return sampling_prob


    def configure_fit(self, server_round: int, parameters: Parameters,
                      client_manager):
        """Sample clients based on pre-determined sampling probabilities"""
        # https://discuss.flower.ai/t/custom-client-selection-strategy/63
        # https://discuss.flower.ai/t/how-do-i-write-a-custom-client-selection-protocol/74/2
        
        # config = {}
        # if self.on_fit_config_fn is not None:
        #     # Custom fit config function provided
        #     config = self.on_fit_config_fn(server_round)
            
            
        # Ensure server_round is included in the config
        config = {"server_round": server_round}  

        if self.on_fit_config_fn is not None:
            # Merge the custom config function's output with the base config
            config.update(self.on_fit_config_fn(server_round))
        
         
        if server_round == 1: 
            # sample all clients for pretraining to estimate G
            selected_clients_cids = list(client_manager.clients.keys())
            
            # Delete the paramaters from the previous run
            config_path = "/home/as1233/incentives/advanced-pytorch/config_utility.json"  
            with open(config_path, "r") as f:
                config_utility = json.load(f)
            for key in ["G_n_squared", "Gn", "cid"]:
                config_utility.pop(key, None) 
            with open(config_path, "w") as file: # save the modified file
                json.dump(config_utility, file, indent=4)
            
            return [(client_manager.clients.get(cid), FitIns(parameters, config)) for cid in selected_clients_cids]  
         
                       
        elif server_round != 1:  
            available_clients = list(client_manager.clients.values()) # clients is dict[cid, ClientProxy]
            selected_clients = [client for client in available_clients if random.random() < self.client_probs[int(client.cid)]]
            selected_clients_cids = [client.cid for client in selected_clients] # This sampling method is common for independent stochastic sampling

            return [(client_manager.clients.get(cid), FitIns(parameters, config)) for cid in selected_clients_cids]
    