# SR respresentation matrix eigendecomposition

### System requirements

* Python3.6
* Tensorflow 1.3

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

