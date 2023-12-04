This repository contains some code for learning a couple of things:

1. The Generative Flow Network (GFlowNet) model.
2. Using Candle and Rust to build and train a deep learning model on a GPU with CUDA to test it out.

As of December 2023, it contains a simple but working prototype for a very basic model. I used it to experiment with the GFlowNet model to understand it better. I may develop this as a proper library or tool for GFlowNet study.

Perform a Test Run
==================

Set up your Rust and CUDA environment (outside the scope of this README, please consult the Internet for that).

Build:



```
cd candle-gfn-bin
cargo install --path ..
```

Run a small testing case:

```
cd example_scripts
bash run_example.sh
```

The results are dump in JSON and SafeTensor format. I will
add a Jupyer Notebook for analyzing the results.

Dec. 2023, Jason