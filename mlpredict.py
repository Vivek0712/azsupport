from azureml.core import Workspace
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core import ScriptRunConfig
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core import Experiment
from azureml.core import Environment
from azureml.core.runconfig import RunConfiguration
from azureml.core.authentication import ServicePrincipalAuthentication
import sys
import time
from azure.storage.queue import QueueClient
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline
import urllib.request
from azureml.core.model import Model
import pathlib
from azure.storage.blob import BlobServiceClient
from azureml.core import Environment
from azureml.core.model import InferenceConfig

from azureml.core import Workspace
from azureml.core.webservice import LocalWebservice
import requests
import json
from azureml.core.compute import ComputeTarget
from azureml.core.compute_target import ComputeTargetException
from azureml.core.compute import AksCompute, ComputeTarget,AmlCompute
from azureml.core.webservice import Webservice, AksWebservice
from azureml.core.webservice import AciWebservice
import requests

import json
import numpy as np
import datetime
import time


def deploy_aci(inputvalues):
    container = inputvalues[0]
    device_id = inputvalues[1]
    expt = inputvalues[2]
    project = inputvalues[3]

    subscription_id = '5b57c18b-0aaf-455e-abb7-4c0474c80bcd'
    resource_group = 'ml'
    workspace_name = 'eegml-gpu'
    ten_id = '38c1e200-3655-4ace-9b0a-d8bec27a9f10'
    service_id = '3d781fd9-66cb-4776-ba03-0b9475d7a765'
    service_pwd = '1~tXhueLl68ugzA.LY_V912LxYKlEw1V~R'
    connect_str = 'DefaultEndpointsProtocol=https;AccountName=eegdatastore;AccountKey=kldfjynCLsfaJcmgSyihYsStwRoLvFYsIMk4hNVBXAYzL7sPP5H1o8oWC3KQpbTDCtGqGMu9kmGsW/STrGgTeA==;EndpointSuffix=core.windows.net'
    
    #Workspace Config

    sp = ServicePrincipalAuthentication(tenant_id=ten_id,  # tenantID
                                        service_principal_id=service_id,  # clientId
                                        service_principal_password=service_pwd)  # clientSecret

    ws = Workspace(subscription_id=subscription_id, resource_group=resource_group, workspace_name=workspace_name,
                       auth=sp)

    modelname = device_id+"_"+expt+"_"+project+"_model"
    try:
        model = Model(ws, name = modelname)

    except:

    # Register model
        model = Model.register(ws, model_name=modelname, model_path="./model.pkl")

    try:
        deploy_env = Environment.get(workspace=ws, name="deployenv")

    except:
        deploy_env = Environment.from_conda_specification("deployenv",pathlib.Path(__file__).parent / "dependencies.yml")
        # deploy_env.docker.base_image = "mcr.microsoft.com/azureml/openmpi3.1.2-cuda10.1-cudnn7-ubuntu18.04"

    inference_config = InferenceConfig(
        environment=deploy_env,
        source_directory=".",
        entry_script="./ssvep.py",
    )


    # Choose a name for your ACI cluster
    aci_name = 'gpucluster3' 

    # Verify that cluster does not exist already
    try:
        aci_target = ComputeTarget(workspace=ws, name=aci_name)
        print('Found existing cluster, use it.')
    except ComputeTargetException:
        # Use the default configuration (can also provide parameters to customize)
        prov_config = AmlCompute.provisioning_configuration("STANDARD_NC12",min_nodes = 4,max_nodes=8)

        # Create the cluster
        aci_target = ComputeTarget.create(workspace = ws, 
                                        name = aci_name, 
                                        provisioning_configuration = prov_config)

    if aci_target.get_status() != "Succeeded":
        aci_target.wait_for_completion(show_output=True)

    deployment_config = AciWebservice.deploy_configuration(cpu_cores = 3.5, memory_gb = 12)

    name = "aciservicemodel4"
    try:
         aci_service =  AciWebservice(ws, name)
    except:

        aci_service = Model.deploy(workspace=ws,
                           name=name,
                           models=[model],
                           inference_config=inference_config,
                           deployment_config=deployment_config,
                           deployment_target = aci_target)
    
    aci_service.wait_for_deployment(show_output = True)
    return aci_service

    
def deploy_local(inputvalues):
    container = inputvalues[0]
    device_id = inputvalues[1]
    expt = inputvalues[2]
    project = inputvalues[3]

    subscription_id = '5b57c18b-0aaf-455e-abb7-4c0474c80bcd'
    resource_group = 'ml'
    workspace_name = 'eegml-gpu'
    ten_id = '38c1e200-3655-4ace-9b0a-d8bec27a9f10'
    service_id = '3d781fd9-66cb-4776-ba03-0b9475d7a765'
    service_pwd = '1~tXhueLl68ugzA.LY_V912LxYKlEw1V~R'
    connect_str = 'DefaultEndpointsProtocol=https;AccountName=eegdatastore;AccountKey=kldfjynCLsfaJcmgSyihYsStwRoLvFYsIMk4hNVBXAYzL7sPP5H1o8oWC3KQpbTDCtGqGMu9kmGsW/STrGgTeA==;EndpointSuffix=core.windows.net'
    
    #Workspace Config

    sp = ServicePrincipalAuthentication(tenant_id=ten_id,  # tenantID
                                        service_principal_id=service_id,  # clientId
                                        service_principal_password=service_pwd)  # clientSecret

    ws = Workspace(subscription_id=subscription_id, resource_group=resource_group, workspace_name=workspace_name,
                       auth=sp)

    modelname = device_id+"_"+expt+"_"+project+"_model"
    try:
        model = Model(ws, name = modelname)

    except:

    # Register model
        model = Model.register(ws, model_name=modelname, model_path="./model.pkl")

    try:
        deploy_env = Environment.get(workspace=ws, name="deployenv")

    except:
        deploy_env = Environment.from_conda_specification("deployenv",pathlib.Path(__file__).parent / "dependencies.yml")
        # deploy_env.docker.base_image = "mcr.microsoft.com/azureml/openmpi3.1.2-cuda10.1-cudnn7-ubuntu18.04"

    inference_config = InferenceConfig(
        environment=deploy_env,
        source_directory=".",
        entry_script="./ssvep.py",
    )
    deployment_config = LocalWebservice.deploy_configuration(port=6789)
    name = "localservicemodel"

    local_service = Model.deploy(workspace=ws,
                           name=name,
                           models=[model],
                           inference_config=inference_config,
                           deployment_config=deployment_config,
                           overwrite=True)
    
    local_service.wait_for_deployment(show_output = True)
    return local_service

def deploy_aks(inputvalues):
    container = inputvalues[0]
    device_id = inputvalues[1]
    expt = inputvalues[2]
    project = inputvalues[3]

    subscription_id = '5b57c18b-0aaf-455e-abb7-4c0474c80bcd'
    resource_group = 'ml'
    workspace_name = 'eegml-gpu'
    ten_id = '38c1e200-3655-4ace-9b0a-d8bec27a9f10'
    service_id = '3d781fd9-66cb-4776-ba03-0b9475d7a765'
    service_pwd = '1~tXhueLl68ugzA.LY_V912LxYKlEw1V~R'
    connect_str = 'DefaultEndpointsProtocol=https;AccountName=eegdatastore;AccountKey=kldfjynCLsfaJcmgSyihYsStwRoLvFYsIMk4hNVBXAYzL7sPP5H1o8oWC3KQpbTDCtGqGMu9kmGsW/STrGgTeA==;EndpointSuffix=core.windows.net'
    
    #Workspace Config

    sp = ServicePrincipalAuthentication(tenant_id=ten_id,  # tenantID
                                        service_principal_id=service_id,  # clientId
                                        service_principal_password=service_pwd)  # clientSecret

    ws = Workspace(subscription_id=subscription_id, resource_group=resource_group, workspace_name=workspace_name,
                       auth=sp)


    modelname = device_id+"_"+expt+"_"+project+"_model"
    try:
        model = Model(ws, name = modelname)

    except:

    # Register model
        model = Model.register(ws, model_name=modelname, model_path="./model.pkl")

    try:
        deploy_env = Environment.get(workspace=ws, name="deployenv")

    except:
        deploy_env = Environment.from_conda_specification("deployenv",pathlib.Path(__file__).parent / "dependencies.yml")
        # deploy_env.docker.base_image = "mcr.microsoft.com/azureml/openmpi3.1.2-cuda10.1-cudnn7-ubuntu18.04"

    inference_config = InferenceConfig(
        environment=deploy_env,
        source_directory=".",
        entry_script="./ssvep.py",
    )


    # Choose a name for your AKS cluster
    aks_name = 'deplyakscluster1' 

    # Verify that cluster does not exist already
    try:
        aks_target = ComputeTarget(workspace=ws, name=aks_name)
        print('Found existing cluster, use it.')
    except:
        # Use the default configuration (can also provide parameters to customize)
        prov_config = AksCompute.provisioning_configuration(vm_size = "Standard_D5_v2")

        # Create the cluster
        aks_target = ComputeTarget.create(workspace = ws, 
                                        name = aks_name, 
                                        provisioning_configuration = prov_config)

        if aks_target.get_status() != "Succeeded":
            aks_target.wait_for_completion(show_output=True)

    deployment_config = AksWebservice.deploy_configuration(cpu_cores = 12, memory_gb = 40,scoring_timeout_ms=1000)    
    name = "aksservicemodel"

    aks_service = Model.deploy(workspace=ws,
                           name=name,
                           models=[model],
                           inference_config=inference_config,
                           deployment_config=deployment_config,
                           deployment_target = aks_target)
    
    aks_service.wait_for_deployment(show_output = True)
    return aks_service  


def runaks():
    ss = deploy_aks(["container1","NEX00000009","exp4","project1"])
    scoring_uri =  ss.scoring_uri
    key1, Key2 = ss.get_keys()
    inp_data = np.random.random((16, 500)).tolist()

    test_sample = json.dumps({'data': inp_data })
    test_sample = bytes(test_sample,encoding = 'utf8')

    headers = {'Content-Type':'application/json', 'Authorization': 'Bearer ' + key1}
    print("AKS \n \n")
    for i in range(10):
        print("Request TS: "+str(datetime.datetime.utcnow()))
        resp = requests.post(scoring_uri, test_sample, headers=headers)
        print("prediction:", resp.text)
        print("Result TS: "+str(datetime.datetime.utcnow()))

        time.sleep(2)

    ss.delete()

def runaci():
    ss = deploy_aci(["container1","NEX00000009","exp4","project1"])
    scoring_uri =  ss.scoring_uri
    # key1, Key2 = ss.get_keys()
    inp_data = np.random.random((16, 500)).tolist()

    test_sample = json.dumps({'data': inp_data })
    test_sample = bytes(test_sample,encoding = 'utf8')

    headers = {'Content-Type':'application/json'}
    # headers = {'Content-Type':'application/json', 'Authorization': 'Bearer ' + key1}
    print("ACI \n \n")
    for i in range(10):
        print("Request TS: "+str(datetime.datetime.utcnow()))
        resp = requests.post(scoring_uri, test_sample, headers=headers)
        print("prediction:", resp.text)
        print("Result TS: "+str(datetime.datetime.utcnow()))

        time.sleep(2)

    ss.delete()

def runlocal():
    ss = deploy_local(["container1","NEX00000009","exp4","project1"])
    scoring_uri =  ss.scoring_uri
    inp_data = np.random.random((16, 500)).tolist()

    test_sample = json.dumps({'data': inp_data })
    test_sample = bytes(test_sample,encoding = 'utf8')
    headers = {'Content-Type':'application/json'}
    print("LOCAL \n \n")
    for i in range(10):
        print("Request TS: "+str(datetime.datetime.utcnow()))
        resp = requests.post(scoring_uri, test_sample, headers=headers)
        print("prediction:", resp.text)
        print("Result TS: "+str(datetime.datetime.utcnow()))

        time.sleep(2)

    ss.delete()



runaks()
# runaci()
# runlocal()
