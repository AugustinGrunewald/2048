import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot import base_bot
from data import bots_name
from modules import playground_bot

import numpy as np
from numpy.typing import NDArray

import random as rd
import copy

from typing import Self

import torch
from torch import nn

import time

import pathlib

class EvolutionaryBotMLP(base_bot.BaseBot):
    def __init__(self, bot_delay : int, model_name_pth : str = "./bot/training/Training_1_/mlp_gen_3_ind_4_.pth") -> None:
        super().__init__(bot_delay)

        # Loading the trained model
        self.__model = torch.load(model_name_pth, weights_only=False)

    def play_move(self, playground : NDArray[np.int64]) -> int:
        prediction : torch.Tensor = self.__model(torch.from_numpy(np.array(playground, dtype=np.float32)).flatten()) # A tensor of shape (1,4)

        return bots_name.movements[int(prediction.argmax().item())]
    
class MultiLayerPerceptron(nn.Module):
    # Adding the error checking to ensure we have the good size of input and output
    def __new__(cls, layers_struct : list[tuple[int, str | None]], index : int, training_id : int) -> Self:
        if layers_struct[0][0] != 16 or layers_struct[-1][0] != 4:
            raise ValueError("The input layer or output layer has not the right size")
        return super().__new__(cls)

    # Passing a layers_struct as parameters, giving the size of each layers and the activation function
    def __init__(self, layers_struct : list[tuple[int, str | None]], index : int, training_id : int) -> None:
        super().__init__()

        # Generation/index of the bot
        self.__generation = 1
        self.__index = index
        self.__layers_struct = layers_struct

        layers = []
        for i in range(len(self.__layers_struct) - 1):
            size_in, _ = self.__layers_struct[i]
            size_out, act_name = self.__layers_struct[i + 1]

            layers.append(nn.Linear(size_in, size_out))
            layers.append(bots_name.activations[act_name]())

        self.__model = nn.Sequential(*layers)
        self.__model.eval() # setting the model to evaluation mode because we're doing evolution and no classical training
 
        # Deactivating the gradient calculation with setting the argument requires_grad to False
        for param in self.__model.parameters():
            param.requires_grad = False

        # Some constants of the number of neuron per layer
        self.__MAX_SIZE = 1024
        self.__MIN_SIZE = 16

        # Constants for the calculation of the note of the bot
        self.__COEF_SCORE = 1/500
        self.__COEF_NUMBER_MVMT = 1/50

        # Outside argument used for training
        self.__training_id = training_id
    # --------------------------------------- #

    @classmethod
    def from_record(cls, saving_name : str) -> 'MultiLayerPerceptron':
        # Getting generation and index : the saving name has always the same format see  @save_model()
        mask = [-7, -4, -2]
        id, gen, ind = np.array(saving_name.split("_"))[mask]

        # Loading the torch model
        model : nn.Sequential = torch.load(saving_name, weights_only=False)
        layers_struct : list[tuple[int, str | None]] = []

        for i, layer in enumerate(model):
            if isinstance(layer, nn.Linear):
                input_size = layer.in_features
                act = None

                if i > 0 and i <= len(model) - 2:
                    act = type(model[i-1]).__name__
                    
                layers_struct.append((input_size, act))

                if i == len(model) - 2: # If we reach the last layer
                    layers_struct.append((layer.out_features, None))

        # Building the new class
        new_mlp : 'MultiLayerPerceptron' = cls(layers_struct, ind, id)
        new_mlp.__generation = gen
        new_mlp.__model = model
        new_mlp.__model.eval() # setting the model to evaluation mode because we're doing evolution and no classical training
 
        # Deactivating the gradient calculation with setting the argument requires_grad to False
        for param in new_mlp.__model.parameters():
            param.requires_grad = False

        return new_mlp 

    @classmethod
    def from_model(cls, mlp : 'MultiLayerPerceptron', new_index : int, weight_bias_mutation : bool = False, structure_mutation : bool = False) -> 'MultiLayerPerceptron':
        # Copying first the structure  
        new_mlp : 'MultiLayerPerceptron' = cls(copy.deepcopy(mlp.__layers_struct), copy.deepcopy(mlp.__index), copy.deepcopy(mlp.__training_id)) if not structure_mutation else cls(mlp.__structure_mutation(), copy.deepcopy(mlp.__index), copy.deepcopy(mlp.__training_id))
        new_mlp.__generation = int(mlp.__generation) + 1 
        new_mlp.__index = new_index

        # Copying the parameters of the model / Taking into acount the decrease or the growth or the absence of changes
        growth = 1 if len(new_mlp.__layers_struct) - len(mlp.__layers_struct) > 0 else (-1 if len(new_mlp.__layers_struct) - len(mlp.__layers_struct) < 0 else 0)
        middle_ind = len(new_mlp.__layers_struct)//2 if growth == 1 else (len(new_mlp.__layers_struct) - 1)//2 # Second case only used when growth == -1

        for name, param in new_mlp.__model.named_parameters(): # Returns only the layers and not the activations layers
            layer_tensor_type = name.split('.')[-1]
            number = int(name.split('.')[-2])

            # Getting the source layer for the current parameter
            layer_ind = (number - 2 if (number >= 2 * middle_ind) else number) if growth == 1 else ((number + 2 if (number > 2 * middle_ind) else number) if growth == -1 else number)
            source_param = mlp.__model[layer_ind]._parameters[layer_tensor_type]

            param.requires_grad = False # Deactivating the gradient calculation with setting the argument requires_grad to False, should be done because copy_ does not change it

            if layer_tensor_type == "bias" and source_param != None: # First changing the bias
                init_size = source_param.size()[0]
                out_size = param.size()[0]

                param[:min(init_size, out_size)].copy_(source_param[:min(init_size, out_size)]) # Always constrained by the minimum input data

            elif layer_tensor_type == "weight" and source_param != None: # Second changing the weight 
                init_size_x, init_size_y = source_param.size()
                out_size_x, out_size_y = param.size()

                param[:min(init_size_x,out_size_x), :min(init_size_y,out_size_y)].copy_(source_param[:min(init_size_x,out_size_x), :min(init_size_y,out_size_y)]) # Always constrained by the minimum input data

        # Modifying the weight and bias if necessary
        if weight_bias_mutation:
            new_mlp.__weight_bias_mutation()

        return new_mlp
        
    # --------------------------------------- #

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        logits = self.__model(x)
        return logits
    
    def save_model(self) -> None:
        saving_name = f"./bot/training/Training_{self.__training_id}_/mlp_gen_{self.__generation}_ind_{self.__index}_.pth"
        pathlib.Path(f"./bot/training/Training_{self.__training_id}_").mkdir(parents=True, exist_ok=True)
        torch.save(self.__model, saving_name)

    def __modify_in_features(self, playground_matrices : NDArray[np.int64]) -> NDArray[np.float32]:
        log2 = False # log2 of each tile
        if log2:
            return np.log2(playground_matrices, dtype=np.float32)
        else: # Without any modification
            return np.array(playground_matrices, dtype=np.float32)
        
    def giving_note(self, performance : dict[str, int]) -> tuple[int, float]:
        # alpha * score - beta * number_movement -> want to penalized to much movements
        return performance["score"], self.__COEF_SCORE * performance["score"] - self.__COEF_NUMBER_MVMT * performance["number_movements"]

    def play(self) -> dict[str, int]:
        playground = playground_bot.Playground()

        while not playground.game_lost() and not playground.frozen_playground():
            prediction : torch.Tensor = self.__model(torch.from_numpy(self.__modify_in_features(playground.get_matrices)).flatten()) # A tensor of shape (1,4)
            model_output = bots_name.movements[int(prediction.argmax().item())]

            playground.update(model_output)

        performance = {
            "score" : playground.score, 
            "number_movements" : playground.number_movements
        }

        return performance

    # --------------------------------------- #

    def __weight_mutation(self, initial_weight : torch.nn.Parameter, sigma : float) -> torch.nn.Parameter: # Note we're not taking only tensors, we consider Parameter for best practices
        gaussian_noise = torch.randn_like(initial_weight) * sigma # We add gaussian noise for the perturbation
        return torch.nn.Parameter(initial_weight + gaussian_noise, requires_grad=False) 

    def __bias_mutation(self, initial_bias : torch.nn.Parameter, sigma : float) -> torch.nn.Parameter: # Should consider a smaller sigma than for the weights
        gaussian_noise = torch.randn_like(initial_bias) * sigma # We add gaussian noise for the perturbation
        return torch.nn.Parameter(initial_bias + gaussian_noise, requires_grad=False)        

    def __weight_bias_mutation(self) -> None:
        for name, param in self.__model.named_parameters():
            if name.split(".")[-1] == "weight":
                param.copy_(self.__weight_mutation(param, 0.025)) # Should be dynamically changed maybe
            else: # we work on the bias
                param.copy_(self.__bias_mutation(param, 0.01)) # Should be dynamically changed maybe

    # --------------------------------------- #
    
    def __activation(self, current_layer : tuple[int, str | None]) -> str | None:
        max_activation_changes = (len(self.__layers_struct) - 1)//2
        if rd.binomialvariate(n=1, p=0.2) == 1 and self.__activation_changes_counter < max_activation_changes: 
            others_activations = [act for act in list(bots_name.activations.keys()) if (act != None and act != current_layer[1])]
            self.__activation_changes_counter += 1
            return rd.choice(others_activations)
        else:
            return current_layer[1] # Note : as we only use this method on inner layers, it never returns None
        
    def __sizes(self, current_layer : tuple[int, str | None]) -> int:
        max_size_changes = (len(self.__layers_struct) - 1)//2
        if rd.binomialvariate(n=1, p=0.6) == 1 and self.__size_changes_counter < max_size_changes: 
            new_size = int(current_layer[0] * (1 + rd.uniform(-0.4, 0.4))) # Changing of +-40%
            self.__size_changes_counter += 1
            return min(max(new_size, self.__MIN_SIZE), self.__MAX_SIZE)
        else:
            return current_layer[0]
        
    def __structure_mutation(self) -> list[tuple[int, str | None]]: 
        output_layers_struct = copy.deepcopy(self.__layers_struct)

        # Keeping the number of inner layers between 1 and 5, low probability chances (5%) and not too many at once, only duplicating layers
        changing_number_layers = True if rd.binomialvariate(n=1, p=0.2) == 1 else False
        number_layers = len(output_layers_struct)

        if changing_number_layers: 
            # +-1 50/50 chance, staying between 1 and 5
            number_layers += 1 if (number_layers == 3) else -1 if (number_layers == 7) else rd.choice([1,-1])
            middle_next_ind = (number_layers + 1)//2     
            
            if number_layers > len(output_layers_struct): # Inserting a layer after the middle or middle+1
                output_layers_struct.insert(middle_next_ind, copy.deepcopy(output_layers_struct[middle_next_ind - 1])) # The copy is unnecessary while using tuple for the layers (which is the case), but never know 
            else: # Removing the layer in the middle or middle+1
                output_layers_struct.pop(middle_next_ind)

        self.__activation_changes_counter = 0
        self.__size_changes_counter = 0

        for i in range(1, number_layers - 1):
            output_layers_struct[i] = (MultiLayerPerceptron.__sizes(self, output_layers_struct[i]), MultiLayerPerceptron.__activation(self, output_layers_struct[i]))
        
        return output_layers_struct



class Population():
    def __init__(self, population_size : int, first_gen_layers_seed : list[tuple[int, str | None]], training_id : int) -> None:
        self.__population_size = population_size
        self.__population = [MultiLayerPerceptron(first_gen_layers_seed, i, training_id) for i in range(self.__population_size)]
        self.__generation = 1
        self.__population_scores : np.ndarray = np.zeros(self.__population_size, dtype=float)
        self.__population_notes : np.ndarray = np.zeros(self.__population_size, dtype=float) 

        self.__NUMBER_GAMES_BOT = 1 # Number of game played by each bot to average its score
        self.__SELECTED_FRACTION = 0.1
        self.__WEIGHT_EVOL_FRACTION = 0.8
        self.__STRUCT_EVOL_FRACTION = 0.2

        self.__last_run_time = 0.

    def __str__(self) -> str:
        # Building output string
        output = "+ ----- ----- DATA ----- ----- +\n+\n"
        output += f"+ Generation n°{self.__generation}\n+\n"
        output += f"+ Population of {self.__population_size}\n+\n"
        output += f"+ Average score : {np.mean(self.__population_scores):.4f} / Average note : {np.mean(self.__population_notes):.4f}\n"
        output += f"+ Median score : {np.median(self.__population_scores):.4f} / Median note : {np.median(self.__population_notes):.4f}\n"
        output += f"+ Max score : {np.max(self.__population_scores):.4f} / Index : {np.argmax(self.__population_scores)} / Max note : {np.max(self.__population_notes):.4f} / Index : {np.argmax(self.__population_notes)}\n"
        output += f"+ Min score : {np.min(self.__population_scores):.4f} / Index : {np.argmin(self.__population_scores)} / Min note : {np.min(self.__population_notes):.4f} / Index : {np.argmin(self.__population_notes)}\n+\n"
        output += f"+ Parameters (see constructor): {self.__NUMBER_GAMES_BOT} / {self.__SELECTED_FRACTION} / {self.__WEIGHT_EVOL_FRACTION} / {self.__STRUCT_EVOL_FRACTION} \n+\n"
        output += f"+ Last evolution elapsed time : {self.__last_run_time:.4f}s\n"
        output += "+ ----- ----- ---- ----- ----- +\n"
        return output
    
    def play(self) -> None:
        # Let each bot play
        for ind, current_bot in enumerate(self.__population):
            score = 0
            note = 0
            for _ in range(self.__NUMBER_GAMES_BOT):
                performance = current_bot.play()
                # Give a note to each bot and saving it in the population_score
                score, note = score + current_bot.giving_note(performance)[0], note + current_bot.giving_note(performance)[1]

            self.__population_scores[ind] = score / self.__NUMBER_GAMES_BOT
            self.__population_notes[ind] = note / self.__NUMBER_GAMES_BOT

    def evolve(self) -> None:
        # Starting the clock
        start = time.perf_counter()

        # Let all the bots play
        self.play()

        # Selecting the bots with the scores
        # Keeping the 10% best scorer and mutate 80% with only the weights and the 20% with also the structure
        top_len = self.__population_size - int(self.__population_size * self.__SELECTED_FRACTION) 
        index = np.argpartition(self.__population_scores, top_len)[top_len:] 
        sorted_index = index[np.argsort(-self.__population_scores[index])] # Sorted from biggest to smallest

        new_population : list[MultiLayerPerceptron] = []
        nmb_weight_mutation_bot = int(self.__WEIGHT_EVOL_FRACTION * self.__population_size // int(self.__population_size * self.__SELECTED_FRACTION))
        nmb_struct_mutation_bot = int(self.__STRUCT_EVOL_FRACTION * self.__population_size // int(self.__population_size * self.__SELECTED_FRACTION))

        counter = 0
        for bot in np.array(self.__population)[sorted_index]:
            for _ in range(nmb_weight_mutation_bot):
                new_population.append(MultiLayerPerceptron.from_model(bot, counter, weight_bias_mutation=True, structure_mutation=False))
                counter += 1
            for _ in range(nmb_struct_mutation_bot):
                new_population.append(MultiLayerPerceptron.from_model(bot, counter, weight_bias_mutation=True, structure_mutation=True))
                counter += 1

        self.__generation += 1
        self.__population = new_population  
        self.__last_run_time = time.perf_counter() - start   

    @property
    def generation(self) -> int:
        return self.__generation

    @generation.setter
    def generation(self, gen : int) -> None:
        self.__generation = gen 

    @property
    def population(self) -> list[MultiLayerPerceptron]:
        return self.__population

    @population.setter
    def population(self, new_population : list[MultiLayerPerceptron]) -> None:
        self.__population = new_population


    def save_population(self) -> None:
        for bot in self.__population:
            bot.save_model()

    @classmethod
    def load_population(cls, training_id : int) -> 'Population':
        directory_path = f"./bot/training/Training_{training_id}_"
        path_object = pathlib.Path(directory_path)

        if path_object.exists():
            sorted_file = sorted([file for file in path_object.iterdir() if file.suffix == ".pth"], key=lambda file : file.name)
            population_size = len(sorted_file)

            loaded_population : list[MultiLayerPerceptron] = [MultiLayerPerceptron.from_record(str(pathlib.Path(directory_path, file.name))) for file in sorted_file]

            new_population : 'Population' = cls(population_size, [(16, None), (1, None), (4, None)], training_id)
            new_population.generation = int(str(sorted_file[0].name).split("_")[-4])
            new_population.population = loaded_population

            return new_population
        
        else:
            raise ValueError("Training ID doesn't exists")
        
if __name__ == "__main__":
    # Initialization of a set of bots
    print("Creating population")
    training_id = 2
    number_bots_per_generation = 250
    first_gen_layers_seed = [(16, None), (32, "ReLU"), (64, "ReLU"), (32, "ReLU"), (4, None)]
    population = Population(number_bots_per_generation, first_gen_layers_seed, training_id)

    # Evolution 
    for epochs in range(1000):
        population.evolve()
        if epochs % 50 == 0:
            print(population)

    population.save_population()

    """# Loading a population from its ID
    print("Loading population")
    try:
        population = Population.load_population(1)
        population.evolve()
        print(population)
    except ValueError as err:
        print(err)"""


