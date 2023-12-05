# generate the speficiation for a random 12x12 grid with random rewards
python generate_spec_random.py > test.json

# train the model from scratch with batch size of 500 trajectories and 320 iteration 
simple_2D_grid_gfn test.json test.1. -b 500 -o 1 -n 320 --save-all-batches

# train the model from the previous trained model `test.2.safetensors` with 
# a batch size of 500 trajectories and 320 iteration and 
# each batch is used for training 5 times
simple_2D_grid_gfn test.json test.2. --model-file test.2.safetensors  -b 500 -o 5 -n 320 --save-all-batches


# simple_2D_grid_gfn test3.json test3 -b 500 -o 1 -n 10000 --save-all-batches
