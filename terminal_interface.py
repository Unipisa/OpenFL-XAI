# Copyright (C) 2023 AI&RD Research Group, Department of Information Engineering, University of Pisa
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
import json


def retrieve_logs(container_name):
    log_file = container_name + ".txt"
    print(log_file)
    Path("./logs").mkdir(parents=True, exist_ok=True)
    os.system(f"docker logs {container_name} > ./logs/{log_file}")
    os.system(f"docker logs {container_name}")


def retrieve_global_model(path, model_name):
    """
    pass folder path to store the global model files
    Format requested:
        
        target_folder
        folder_parent/target_folder
        ...

    Starting folder: global_models

    It is not necessary to have the folder structure prepared in advance,
    the script will create what is missing in the file system
        
    Three files will be saved in .npy format inside target_folder:
        rules   --    TSK_global_rules.npy
        weights --    TSK_global_weights.npy   
    """
    path = "./global_models/"+path
    Path(path).mkdir(parents=True, exist_ok=True)

    os.system("docker cp aggregator_xai:/current_workspace/"+model_name+"_global_model_rules_antec.npy "+path+"/"+model_name+"_global_model_rules_antec.npy")
    os.system("docker cp aggregator_xai:/current_workspace/"+model_name+"_global_model_rules_conseq.npy "+path+"/"+model_name+"_global_model_rules_conseq.npy")
    os.system("docker cp aggregator_xai:/current_workspace/"+model_name+"_global_model_weights.npy "+path+"/"+model_name+"_global_model_weights.npy")


def rebuild_images():
    
    if os.system("docker image inspect openfl_xai > /dev/null 2>&1") == 256:
        os.system("docker build -f Dockerfile.openfl_xai -t openfl_xai .") 

    # remove all containers
    os.system("docker rm -f $(docker ps -aq)")

    # remove images
    os.system("docker rmi openfl_xai/aggregator openfl_xai/collaborator")

    # build collaborators images
    os.system("docker build -f Dockerfile.xai_collaborator -t openfl_xai/collaborator .")

    # build aggregator image
    os.system("docker build -f Dockerfile.xai_aggregator -t openfl_xai/aggregator .")

    return


def start_new_federation():
    os.system("docker rm -f $(docker ps -aq)")
    os.system("docker compose up -d")


def print_help():
    # make a txt file with instructions and read - print it
    os.system("clear")
    print("---------------------------")
    print("build:   execute the docker build command to generate Docker images for AggregatorXAI and CollaboratorXAI components")
    print("start:   instantiate the containers as specified in docker-compose.yml. Upon creation, either containers will start AggregatorXAI or CollaboratorXAI instance")
    print("status:  execute command docker ps -aq to show all existing containers and their status onto the system")
    print("logs:    asks in input a container name, then print the output of the specified container on screen and into .txt file under /logs directory ")
    print("save:    intended use only after all contaienrs have terminated their execution without errors. \
             Ask for folder name, then download under ./global_models/folder the aggregated model")
    print("cls:     clear screen output")
    print("quit:    terminate interface execution")
    print("further information on the usage of the example can be found at: https://github.com/MattiaDaole/OpenFL-XAI#illustrative-example")
    input("press enter to exit")


if __name__ == "__main__":
    os.system("clear")

    with open("./configuration.json", "r") as f:
        config = json.load(f)
    print("Configuration Imported:")
    print(config)

    while True:
        print("---------------------------")
        print("Command Aliases")
        print("build    -   Build Images")
        print("start    -   Start new Federation")
        print("status   -   Check Containers status")
        print("logs     -   Retrieve Container Logs")
        print("save     -   Retrieve Global Model")
        print("cls      -   Clear Interface Output")
        print("help     -   Additional information about interface's commands")
        print("quit     -   Quit")
        print("---------------------------")

        choice = input("Command: ")

        if choice == "build":
            rebuild_images()
        elif choice == "start":
            start_new_federation()
        elif choice == "status":
            os.system("docker ps -a")
        elif choice == "logs":
            container_name = input("Specify container name (examples: aggregator-xai, col0 ): ")
            retrieve_logs(container_name)
        elif choice == "save":
            path = input("Specify Path to store model (no white spaces allowed): ")
            retrieve_global_model(path, config["model_name"])
        elif choice == "cls":
            os.system("clear")
        elif choice == "help":
            print_help()
        elif choice == "quit":
            os.system("clear")
            break
        else:
            print("Wrong Input")
            continue


