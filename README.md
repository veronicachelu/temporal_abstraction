## Direction-based Option Critic with Successor Representations 

Intelligent behavior manifests in animals at different levels of granularity over the temporal scale (e.g motor movements can last a few seconds; by contrast, a strategy can be extended over hours, days or even years as life resolutions). 
Historically, the prime focus has been on agents that reason at the smallest granularity level, where they only perform the primitive actions constrained by the environment, without the ability to explore a spectrum of time-scales and the capacity to hierarchically build along the abstraction dimension. 

Hierarchical Reinforcement Learning [1], inspired by human-like behaviour of reasoning over prolonged periods of time and at multiple levels of detail, is a key ingredient in scaling up the framework of general purpose learning to complex behaviors. Temporally abstracting behaviour allows for more sample efficient exploration, faster learning, planning and inductive transfer, especially in terms of an unstructured and very sparse signal from the environment. 

Intrinsic motivation is believed to by essential in the automatic decomposition of a task into a hierarchy, however option discovery and subgoal identification is still a challenge and the exact objective for encouraging abstraction is an open issue. Using a low-dimensional eigendecomposition of the successor representations of states encountered by an agent in an environment [2] can be seen as a way to use intrinsic motivation to discover temporally abstract actions that explore the state manifold. Here, we investigate combining the eigenoptions discovery with the option-critic framework [3] in order to learn options that explore the environment at the same time with a policy over options that optimizes for the goal.

Using a spectral eigendecomposition of the successor representations of states [2] can be seen as a way to use intrinsic motivation to discover temporally abstract actions. In this report, we investigate combining the eigenoptions discovery with  the option-critic framework [3] in order to learn options that explore the environment at the same time with a policy over options that optimizes for the goal. 

### Brief

This repository features a potential strategy for using the geometry of the environment for exploration with temporally abstracted actions while at the same time optimizing for the extrinsic reward signal received from the environment. 

I ran some some experiments with linear function approximation over one-hot states and non-linear function approximation over high-dimensional input states (pixel space) in the 4-Rooms environment illustrating the properties of the singular decomposition of the SR matrix. The features are learned by next frame prediction from pixels and the SR by TD-learning with multi-threaded asyncronous agents.

Then there are experiments using the Option-Critic framework in the same environment in comparison with the direction-based agent. The direction-based agent architecture uses exploration by means of the spectral decomposition of the SR matrix into useful traversal directions over the state manifold. 
 
I have also performed experiments in which the options are no longer discreate, but direction embeddings. However, in this case the experiments point toward the fact that the agent learn to ignore the high-level directions.

In the continual learning scenario, every 1000 episodes the goal position is changed. The figures illustrate the learning 
curves of both agents in comparison.


#### System requirements

* Python3.6
* Tensorflow 1.4

        pip install -r requirements.txt 

### Successor Features respresentation decomposition with ONE-HOT states and linear function approximation

#### Training, resuming & plotting

* To train SR representation weights use:

        python train.py --logdir=./logdir --config=linear_sf --task="train" --resume=False

* To resume training SR representation weights use:

        python train.py --logdir=./logdir --config=linear_sf --task=train --resume=True
        
* To plot SR_vectors, SR_matrix, EigenVectors of the SR matrix, the EigenValues of the SR_matrix
 and the learned value function and policy coresponding to each eigenvector use:
        
        python train.py --logdir=./logdir --config=linear_sf --task=build_sr_matrix --resume=True --load_from=./logdir/0-linear_sf
        
* To see training progress run tensorboard from the ```logdir/0-linear_sf/summaries``` directory:
       
       tenorboard --logdir=.

* After running the plot command the eigenvalues plot is in ```logdir/0-linear_sf```.
* The sr vectors, eigenvectors, value function and policy learned are in ```logdir/0-linear_sf/summaries```.

#### Training Results```````````````````````````````````````````````````````

https://drive.google.com/drive/folders/0B_qT_xcPy4w3N1VoYmhBWlhadEk?usp=sharing

### Successor Features respresentation decomposition with high-dimensional pixel input states and non-linear function approximation 
* trained with multiple asyncronous agents

#### Training, resuming & plotting

* To train SR representation weights use:

        python train.py --logdir=./logdir --config=dynamic_SR --task=train --resume=False

* To resume training SR representation weights use:

        python train.py --logdir=./logdir --config=dynamic_SR --task=train --resume=True --load_from=<dir_to_load_from>
        
* To plot SR_vectors, SR_matrix, EigenVectors of the SR matrix, the EigenValues of the SR_matrix
 and the learned value function and policy coresponding to each eigenvector use:
        
        python train.py --logdir=./logdir --config=dynamic_SR --task=build_sr_matrix --resume=True --load_from=<dir_to_load_from>
        
        
* To see training progress run tensorboard from the ```logdir/<logdir_dynamic_SR_dir>/dif/summaries``` directory:
       
       tenorboard --logdir=.


#### Training Results```````````````````````````````````````````````````````

https://drive.google.com/open?id=10PDUOyclVthsLwW9uq8nZmUF6Suj2d5A


### Option Critic (OC)


#### Training, resuming & plotting

* To train OC use:

        python train.py --logdir=./logdir --config=oc --task=train --resume=False

* To resume training OC use:

        python train.py --logdir=./logdir --config=oc --task=train --resume=True --load_from=<dir_to_load_from>
        
* To see training progress run tensorboard from the ```logdir/<logdir_oc_dir>/dif/summaries``` directory:
       
       tenorboard --logdir=.
       
* To see clips of the agent's performance in each episode and the results of all the eval episodes go to ```logdir/<logdir_oc_dir>/dif/test``` directory
       

### Direction-based Option Critic with exploration based on eigendecomposition of the Successor Representation matrix - as a means to explore the state-space manifold

### Training, resuming & plotting

* To train direction-based OC use:

        python train.py --logdir=./logdir --config=eigenoc_dyn --task=train --resume=False

* To resume training of the direction-based OC use:

        python train.py --logdir=./logdir --config=eigenoc_dyn --task=train --resume=True --load_from=<dir_to_load_from>
        

* To see training progress run tensorboard from the ```logdir/<logdir_oc_dir>/dif/summaries``` directory:
       
       tenorboard --logdir=.
       

### References

[1] [Between MDPs and semi-MDPs: A Framework for Temporal Abstraction in Reinforcement Learning - Sutton, Richard S. and Precup, Doina and Singh, Satinder] (http://dx.doi.org/10.1016/S0004-3702(99)00052-1)
[2] [[Eigenoption Discovery through the Deep Successor Representation - Marlos C. Machado, Clemens Rosenbaum, Xiaoxiao Guo, Miao Liu, Gerald Tesauro, Murray Campbell]](https://arxiv.org/abs/1710.11089)
[3] [[The Option-Critic Architecture - Pierre-Luc Bacon, Jean Harb, Doina Precup]](http://arxiv.org/abs/1609.05140)
