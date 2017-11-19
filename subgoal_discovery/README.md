# SF respresentation matrix eigendecomposition

### System requirements

* Python3.6
* Tensorflow 1.3

### Training, resuming & plotting

* To train SF representation weights use:

        python train_DIF.py --logdir=./logdir --config=linear_4rooms --task="sf" --resume=False

* To resume training SR representation weights use:

        python train_DIF.py --logdir=./logdir --config=linear_4rooms --task="sf" --resume=True

* To see training progress run tensorboard from the ```logdir/linear_sf/summaries``` directory:
       
       tenorboard --logdir=.
               