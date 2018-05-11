# EigenOption Critic with Successor Representations (EOC-SR)

Hierarchical Reinfocement Learning is a key ingredient in scaling up the framework of general 
purpose learning to complex behaviors. This is motivated by our understanding of human 
reasoning at higher levels then mere primitive actions. Temporal abstraction leads to faster 
learning and inductive transfer, especially in terms of an unstructured and very sparse signal
 from the environment. Intrinsic motivation is believed to by essential to automatic decomposition
  of a task into a hierarchy, however option discovery and subgoal identification using function
   approximation is still a challenge in high dimensional MDPs with unstructured reward signals. 
   Using a low-dimensional eigendecomposition of the successor representations of states 
   [1] can be seen as a way to use intrinsic motivation to discover temporally 
   abstract actions. In this report, we investigate combining the eigenoptions discovery with 
   the option-critic framework [2] in order to learn options that explore the environment 
   at the same time with a policy over options that optimizes for the goal. 
### Brief

The idea of using the successor feature representation as a predictive model for encoding trajectories of information flow  is even more challenging in the presence of function approximation. Real world environments are very large in nature and their states cannot be enumerated or represented uniquely. As a result, this requires distributed feature representations using neural networks. To be able to correctly construct successor features representations, the network also needs to model a reconstruction loss of the next frame prediction as an auxiliary task. Finally, being able to find interesting options is not enough, since the agent's goal will usual be specified by a sparse reward structure which it must use correspondingly by optimizing the option sequence to achieve the goal.

This is a potential strategy for using the geometry of the environment for exploration with temporally abstracted actions while at the same time optimizing for the extrinsic reward signal received from the environment. I ran some some experiments with one-hot states and non-linear functional approximation in the 4-Rooms environment illustrating the properties of the low level decomposition of the SR matrix. The features are learned by next frame prediction from pixels and the SR by TD-learning multi-threaded asyncronous and also with a buffer.

Then there are experiments using the Option-Critic framework in the same environment in comparison with the EigenOption-Criti-SR aagent. The EigenOption-Critic with SR architecture which uses exploration by means of the low-level decomposition of the SR matrix into useful traversal directions over the state manifold. 

Every 1000 episodes the goal position is changed. The figures illustrate the learning curves of both agents in comparison.

### Results

![Alt text](images/1.png?raw=true "Agent training" )
![Alt text](images/2.png?raw=true "Agent training")

### System requirements

* Python3.6
* Tensorflow 1.4

        pip install -r requirements.txt 

### Training, resuming & plotting

* To train EOC-SR for Montezuma's Revenge use:

        python train.py --logdir=./logdir --config=eigenoc_montezuma --task=sf --resume=False --num_agents=32 --nb_options=8

* To resume training EOC-SR use:

        python train.py --logdir=./logdir --config=eigenoc_montezuma --task=sf --resume=True --load_from=<dir_to_load_from> --num_agents=32 --nb_options=8
        
* To eval EOC-SR use:
                
        python train.py --logdir=./logdir --config=eigenoc_montezuma --task=eval --resume=True --load_from=<dir_to_load_from> --num_agents=32 --nb_options=8
        
* To see training progress run tensorboard from the ```logdir/<logdir_oc_dir>/dif/summaries``` directory:
       
       tenorboard --logdir=.
       
* To see clips of the agent's performance in each episode and the results of all the eval episodes go to ```logdir/<logdir_oc_dir>/dif/test``` directory
       

* To train EOC-SR use:

        python train.py --logdir=./logdir --config=eigenoc_dyn --task=sf --resume=False

* To resume training EOC-SR use:

        python train_DIF.py --logdir=./logdir --config=eigenoc_dyn --task=sf --resume=True --load_from=<dir_to_load_from>
        
* To eval EOC-SR use:
                
        python train.py --logdir=./logdir --config=eigenoc_dyn --task=eval --resume=True --load_from=<dir_to_load_from>
        
* To see training progress run tensorboard from the ```logdir/<logdir_oc_dir>/dif/summaries``` directory:
       
       tenorboard --logdir=.
       
* To see clips of the agent's performance in each episode and the results of all the eval episodes go to ```logdir/<logdir_oc_dir>/dif/test``` directory
       

# Option Critic (OC)

### System requirements

* Python3.6
* Tensorflow 1.4

### Training, resuming & plotting

* To train OC use:

        python train.py --logdir=./logdir --config=oc_dyn --task=sf --resume=False

* To resume training OC use:

        python train.py --logdir=./logdir --config=oc_dyn --task=sf --resume=True --load_from=<dir_to_load_from>
        
* To eval OC use:
                
        python train.py --logdir=./logdir --config=oc_dyn --task=eval --resume=True --load_from=<dir_to_load_from>
        
* To see training progress run tensorboard from the ```logdir/<logdir_oc_dir>/dif/summaries``` directory:
       
       tenorboard --logdir=.
       
* To see clips of the agent's performance in each episode and the results of all the eval episodes go to ```logdir/<logdir_oc_dir>/dif/test``` directory
       

# SR respresentation matrix eigendecomposition with NN and Asyncronous training (A3C)

### System requirements

* Python3.6
* Tensorflow 1.4

### Training, resuming & plotting

* To train SR representation weights use:

        python train.py --logdir=./logdir --config=dynamic_SR --task=sf --resume=False

* To resume training SR representation weights use:

        python train.py --logdir=./logdir --config=dynamic_SR --task=sf --resume=True --load_from=<dir_to_load_from>
        
* To plot SR_vectors, SR_matrix, EigenVectors of the SR matrix, the EigenValues of the SR_matrix
 and the learned value function and policy coresponding to each eigenvector use:
        
        python train.py --logdir=./logdir --config=dynamic_SR --task=matrix --resume=True --load_from=<dir_to_load_from>
        
        
* To see training progress run tensorboard from the ```logdir/<logdir_dynamic_SR_dir>/dif/summaries``` directory:
       
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

        python train.py --logdir=./logdir --config=linear_sf --task="sf" --resume=False

* To resume training SR representation weights use:

        python train.py --logdir=./logdir --config=linear_sf --task="sf" --resume=True
        
* To plot SR_vectors, SR_matrix, EigenVectors of the SR matrix, the EigenValues of the SR_matrix
 and the learned value function and policy coresponding to each eigenvector use:
        
        python train.py --logdir=./logdir --config=linear_sf --task="matrix" --resume=True
        
* To see training progress run tensorboard from the ```logdir/linear_sf/summaries``` directory:
       
       tenorboard --logdir=.

* After running the plot command the eigenvalues plot is in ```logdir/linear_sf```.
* The eigenvectors, value function and policy learned are in ```logdir/linear_sf/summaries```.
* The sr_vectors plotted on the game environment are in  ```logdir/linear_sf/summaries/sr_vectors```

### Training Results```````````````````````````````````````````````````````

https://drive.google.com/drive/folders/0B_qT_xcPy4w3N1VoYmhBWlhadEk?usp=sharing


### References

[1] [[Eigenoption Discovery through the Deep Successor Representation - Marlos C. Machado, Clemens Rosenbaum, Xiaoxiao Guo, Miao Liu, Gerald Tesauro, Murray Campbell]](https://arxiv.org/abs/1710.11089)
[2] [[The Option-Critic Architecture - Pierre-Luc Bacon, Jean Harb, Doina Precup]](http://arxiv.org/abs/1609.05140)