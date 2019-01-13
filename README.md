docker run -v /path/to/json/files/:/working_dir/metadata -v /path/to/image/files/:/working_dir/images --shm-size 50G -it kodakone_challenge /bin/bash

python3 generate_dataset.py
python3 main.py
