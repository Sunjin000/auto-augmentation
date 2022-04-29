from dataclasses import dataclass
from flask import Flask, request, current_app, send_file, send_from_directory, redirect, url_for, session
from flask_cors import CORS, cross_origin
import os
import zipfile
import torch
from numpy import int0, save, load
from react_backend.wapp_util import parse_users_learner_spec
import pprint
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
torch.manual_seed(0)

print('@@@ import successful')

# app = Flask(__name__, static_folder='react_frontend/build', static_url_path='/')
app = Flask(__name__)
CORS(app)

# it is used to collect user input and store them in the app
@app.route('/home', methods=["GET", "POST"])
# @cross_origin()
def get_form_data():
    
    if request.method == 'POST':
        print('@@@ in Flask Home')
        
        form_data = request.form
        print('@@@ this is form data', form_data)

        # required input
        ds = form_data['select_dataset'] # pick dataset (MNIST, KMNIST, FashionMNIST, CIFAR10, CIFAR100)
        IsLeNet = form_data["select_network"]   # using LeNet or EasyNet or SimpleNet ->> default 
        auto_aug_learner = form_data["select_learner"] # augmentation methods to be excluded

        print('@@@ required user input:', 'ds', ds, 'IsLeNet:', IsLeNet, 'auto_aug_leanrer:',auto_aug_learner)
        # advanced input
        if form_data['batch_size'] not in ['undefined', ""]: 
            batch_size = int(form_data['batch_size']    )   # size of batch the inner NN is trained with
        else: 
            batch_size = 16 # this is for demonstration purposes
        if form_data['learning_rate'] not in ['undefined', ""]: 
            learning_rate =  float(form_data['learning_rate'])  # fix learning rate
        else: 
            learning_rate = 1e-2
        if form_data['toy_size'] not in ['undefined', ""]: 
            toy_size = float(form_data['toy_size'])      # total propeortion of training and test set we use
        else: 
            toy_size = 0.01 # this is for demonstration purposes
        if form_data['iterations'] not in ['undefined', ""]: 
            iterations = int(form_data['iterations'])      # total iterations, should be more than the number of policies
        else: 
            iterations = 2
        exclude_method = form_data['select_action']
        print('@@@ advanced search: batch_size:', batch_size, 'learning_rate:', learning_rate, 'toy_size:', toy_size, 'iterations:', iterations, 'exclude_method', exclude_method)
        

        # default values 
        max_epochs = 10      # max number of epochs that is run if early stopping is not hit
        early_stop_num = 10   # max number of worse validation scores before early stopping is triggered
        num_policies = 5      # fix number of policies
        num_sub_policies = 5  # fix number of sub-policies in a policy
        
        
        # if user upload datasets and networks, save them in the database
        if ds == 'Other':
            ds_folder = request.files['ds_upload'] 
            ds_name_zip = ds_folder.filename
            # check dataset zip file format
            if ds_name_zip.split('.')[1] != 'zip':
                data = {'error_type': 'not a zip file', 'error': "We found that your uplaoded dataset is not a zip file..."}
                current_app.config['data'] = data
                return data
            ds_name = ds_name_zip.split('.')[0]
            ds_folder.save('./react_backend/datasets/'+ ds_name_zip)
            with zipfile.ZipFile('./react_backend/datasets/'+ ds_name_zip, 'r') as zip_ref:
                zip_ref.extractall('./react_backend/datasets/upload_dataset/')
            if not current_app.debug:
                os.remove(f'./react_backend/datasets/{ds_name_zip}')
        else: 
            ds_name_zip = None
            ds_name = None

        # test if uploaded dataset meets the criteria 
        i = -1
        folder = 0
        for (dirpath, dirnames, filenames) in os.walk(f'./react_backend/datasets/upload_dataset/{ds_name}/'):
            i += 1
            if i==0:
                folders = dirnames
            has_child_folder = dirnames!=[] # check if there are child folders
            if not has_child_folder and i==0: 
                data = {'error_type': 'incorret dataset', 
                        'error': "We found that your uplaoded dataset doesn't have the correct format that we are looking for."}
                current_app.config['data'] = data
                return data
        if  folder!=0 and len(folders)!=i:
            data = {'error_type': 'incorret dataset', 
                    'error': "We found that your uplaoded dataset doesn't have the correct format that we are looking for."}
            current_app.config['data'] = data
            return data
        print('@@@ correct dataset folder!')
        
        # save the user uploaded network
        if IsLeNet == 'Other':
            childnetwork = request.files['network_upload']
            network_name = childnetwork.filename
            if network_name.split('.')[1] != 'pkl':
                data = {'error_type': 'incorrect network', 
                        'error': "We found that your uploaded network is not a pickle file"}
                current_app.config['data'] = data
                return data
            else: 
                childnetwork.save('./child_networks/'+childnetwork.filename)
        else: 
            network_name = None

        print("@@@ user input has all stored in the app")

        data = {'ds': ds, 'ds_name': ds_name_zip, 'IsLeNet': IsLeNet, 'network_name': network_name,
                'auto_aug_learner':auto_aug_learner, 'batch_size': batch_size, 'learning_rate': learning_rate, 
                'toy_size':toy_size, 'iterations':iterations, 'exclude_method': exclude_method, }

        current_app.config['data'] = data
        
        print('@@@ all data sent', current_app.config['data'])


    elif request.method == 'GET':
        print('it is GET method')
    
        if 'data' in current_app.config.keys():
            data = current_app.config['data']
        else: 
            data = {'error': "We didn't received any data from you submission form. Please go back to the home page", 
            'error_type': 'no data'}

    return data
    # return redirect(url_for('confirm', data=data))



# ========================================================================
@app.route('/training', methods=['POST', 'GET'])
@cross_origin()
def training():

    # default values 
    max_epochs = 10      # max number of epochs that is run if early stopping is not hit
    early_stop_num = 10   # max number of worse validation scores before early stopping is triggered
    num_policies = 5      # fix number of policies
    num_sub_policies = 5  # fix number of sub-policies in a policy
    data = current_app.config.get('data')

    # parse the settings given by the user to obtain tools we need
    train_dataset, test_dataset, child_archi, agent = parse_users_learner_spec(
                                            max_epochs=max_epochs,
                                            early_stop_num=early_stop_num,
                                            num_policies=num_policies,
                                            num_sub_policies=num_sub_policies,
                                            **data
                                        )

    # train the autoaugment learner for number of `iterations`
    agent.learn(
        train_dataset=train_dataset, 
        test_dataset=test_dataset, 
        child_network_architecture=child_archi,
        iterations=data['iterations']
        )
    
    print('the history of all the policies the agent has tested:')
    pprint.pprint(agent.history)

    # get acc graph and best acc graph
    acc_list = [acc for (policy,acc) in agent.history]
    best_acc_list = []
    best_til_now = 0
    for acc in acc_list:
        if acc>best_til_now:
            best_til_now=acc
        best_acc_list.append(best_til_now)
    
    # plot both here
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(acc_list)
    ax.plot(best_acc_list)
    ax.set_xlabel('Number of Iterations')
    ax.set_ylabel('Accuracy')
    ax.set_title('Auto-augmentation Learner Performance Curve')
    with open("./react_frontend/src/pages/output.png", 'wb') as f:
        fig.savefig(f)

    print("best policies:")
    best_policy = agent.get_mega_policy(number_policies=4)
    print(best_policy)
    with open("./react_backend/policy.txt", 'w') as f:
        # save the best_policy in pretty_print string format
        f.write(pprint.pformat(best_policy, indent=4))

    print('')

    return {'status': 'Training is done!'}


# ========================================================================
@app.route('/result')
@cross_origin()
def show_result():
    file_path = "./react_backend/policy.txt"
    f = open(file_path, "r")
    return send_file(file_path, as_attachment=True)



# @app.route('/')
# def serve():
#     return send_from_directory(app.static_folder, 'index.html')



if __name__ == '__main__':
    app.run(debug=False, use_reloader=False)