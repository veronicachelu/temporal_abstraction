# EigenOption Critic (EOC)

### System requirements

* Python3.6
* Tensorflow 1.4

### Training, resuming & plotting

* To train EOC use:

        python train_DIF.py --logdir=./logdir --config=eigenoc --task=sf --resume=False

* To resume training EOC use:

        python train_DIF.py --logdir=./logdir --config=eigenoc --task=sf --resume=True --load_from=<dir_to_load_from>
        
* To eval EOC use:
                
        python train_DIF.py --logdir=./logdir --config=eigenoc --task=eval --resume=True --load_from=<dir_to_load_from>
        
* To see training progress run tensorboard from the ```logdir/<logdir_oc_dir>/dif/summaries``` directory:
       
       tenorboard --logdir=.
       
* To see clips of the agent's performance in each episode and the results of all the eval episodes go to ```logdir/<logdir_oc_dir>/dif/test``` directory
       


### Evaluation Results```````````````````````````````````````````````````````

* Goal changes location every 1000 episodes

![Alt text](images/1.png?raw=true "Agent training" )
![Alt text](images/2.png?raw=true "Agent training")


# Option Critic (OC)

### System requirements

* Python3.6
* Tensorflow 1.4

### Training, resuming & plotting

* To train OC use:

        python train_DIF.py --logdir=./logdir --config=oc --task=sf --resume=False

* To resume training OC use:

        python train_DIF.py --logdir=./logdir --config=oc --task=sf --resume=True --load_from=<dir_to_load_from>
        
* To eval OC use:
                
        python train_DIF.py --logdir=./logdir --config=oc --task=eval --resume=True --load_from=<dir_to_load_from>
        
* To see training progress run tensorboard from the ```logdir/<logdir_oc_dir>/dif/summaries``` directory:
       
       tenorboard --logdir=.
       
* To see clips of the agent's performance in each episode and the results of all the eval episodes go to ```logdir/<logdir_oc_dir>/dif/test``` directory
       

Trained model: loading...

# SR respresentation matrix eigendecomposition with NN and Asyncronous training (A3C)

### System requirements

* Python3.6
* Tensorflow 1.4

### Training, resuming & plotting

* To train SR representation weights use:

        python train_DIF.py --logdir=./logdir --config=dif_4rooms_fc --task=sf --resume=False

* To resume training SR representation weights use:

        python train_DIF.py --logdir=./logdir --config=dif_4rooms_fc --task=sf --resume=True --load_from=<dir_to_load_from>
        
* To plot SR_vectors, SR_matrix, EigenVectors of the SR matrix, the EigenValues of the SR_matrix
 and the learned value function and policy coresponding to each eigenvector use:
        
        python train_DIF.py --logdir=./logdir --config=dif_4rooms_fc --task=matrix --resume=True --load_from=<dir_to_load_from>
        
        
* To see training progress run tensorboard from the ```logdir/<logdir_dif_4rooms_fc_dir>/dif/summaries``` directory:
       
       tenorboard --logdir=.


### Training Results```````````````````````````````````````````````````````

https://drive.google.com/open?id=10PDUOyclVthsLwW9uq8nZmUF6Suj2d5A

# SR respresentation matrix eigendecomposition with NN and DQN

### System requirements

* Python3.6
* Tensorflow 1.4

### Training, resuming & plotting

* To train SR representation weights use:

        python subgoal_discovery/train.py --logdir=./subgoal_discovery/logdir --config=dqn_sf_4rooms_fc2 --task=sf --resume=False

* To resume training SR representation weights use:

        python subgoal_discovery/train.py --logdir=./subgoal_discovery/logdir --config=dqn_sf_4rooms_fc2 --task=sf --resume=True --load_from=<dir_to_load_from>
        
* To plot SR_vectors, SR_matrix, EigenVectors of the SR matrix, the EigenValues of the SR_matrix
 and the learned value function and policy coresponding to each eigenvector use:
        
        python subgoal_discovery/train.py --logdir=./subgoal_discovery/logdir --config=dqn_sf_4rooms_fc2 --task=matrix --resume=True --load_from=<dir_to_load_from>
        
        
* To see training progress run tensorboard from the ```subgoal_discovery/logdir/<logdir_dqn_sf_4rooms_fc2_dir>/dif/summaries``` directory:
       
       tenorboard --logdir=.


### Training Results```````````````````````````````````````````````````````

https://drive.google.com/open?id=1cVNgB-VZrob31ZXsZJjn28htcAD2xviq

# SR respresentation matrix eigendecomposition with ONE-HOT

### System requirements

* Python3.6
* Tensorflow 1.4

### Training, resuming & plotting

* To train SR representation weights use:

        python train_linear_sf.py --logdir=./logdir --config=linear_4rooms --task="sf" --resume=False

* To resume training SR representation weights use:

        python train_linear_sf.py --logdir=./logdir --config=linear_4rooms --task="sf" --resume=True
        
* To plot SR_vectors, SR_matrix, EigenVectors of the SR matrix, the EigenValues of the SR_matrix
 and the learned value function and policy coresponding to each eigenvector use:
        
        python train_linear_sf.py --logdir=./logdir --config=linear_4rooms --task="matrix" --resume=True
        
* To see training progress run tensorboard from the ```logdir/linear_sf/summaries``` directory:
       
       tenorboard --logdir=.

* After running the plot command the eigenvalues plot is in ```logdir/linear_sf```.
* The eigenvectors, value function and policy learned are in ```logdir/linear_sf/summaries```.
* The sr_vectors plotted on the game environment are in  ```logdir/linear_sf/summaries/sr_vectors```

### Training Results```````````````````````````````````````````````````````

https://drive.google.com/drive/folders/0B_qT_xcPy4w3N1VoYmhBWlhadEk?usp=sharing

