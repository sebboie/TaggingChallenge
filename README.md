# README.md

## The image can be obtained from Dockerhub via:

docker pull sebboie/taggingchallenge

#### The container can be run by issuing:
docker run -v /path/to/json/files/:/working_dir/metadata -v /path/to/image/files/:/working_dir/images --shm-size 50G -it kodakone_challenge /bin/bash

*/path/to/json/files/* refers to your local path where json files (e.g. yelp_academic_dataset_business.json, ...) are stored
*/path/to/image/files/* refers to your local path where the images are stored

#### generates labels and corresponding image csv's for training and testing
python3 generate_dataset.py 

#### Run training and testing
python3 main.py 
