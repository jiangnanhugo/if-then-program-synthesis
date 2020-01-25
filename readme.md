## Experiment code for If-Then Program Synthesis
### Environment
- python 3
- tensorflow 1.13.1
- numpy
- tqdm

### model configurations
in the folder `configs/`.
we have provide two configurations for medium and large scale models.

### run the model
in the folder `bash_dir`.
We have provided the bash script for runing on two different dataset with different CRISP width.


### datasets
in the folder `dataset/`.
- IFTTT are based on paper from [latent attention for if-then program synthesis].
- Zapier are directly collected from `zapier.com` websites.

### results
in the `model/` folder. we include results of two datasets in terms of two metric.

### implementation of MDD
it is in the `tf_utils/mdd` folder. We include the implementation of `arc` and `state node` and `build mdd` in corresponding files.
