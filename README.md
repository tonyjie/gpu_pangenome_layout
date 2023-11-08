# Rapid GPU-Based Pangenome Graph Layout

This project is built upon [ODGI](https://github.com/pangenome/odgi) (optimized dynamic genome/graph implementation), a comprehensive pangenome analysis framework. 
The original ODGI only supports multi-threaded CPU implementation. 

Among all the supported operations within ODGI, [`odgi layout`](https://odgi.readthedocs.io/en/latest/rst/commands/odgi_layout.html) is the most time-consuming step, which becomes the bottleneck of the pangenome analysis pipeline. 

We provide a rapid GPU solution for pangenome graph layout, leading to a speedup of 48.3x, reducing the previous hour-scale computation time into only minute-scale computation time. 

## Installation
### Directly use it in the upstream [ODGI](https://github.com/pangenome/odgi)
- We are actively working on pushing the PR to the upstream ODGI! Then you can directly install and use it in the ODGI framework (very soon). 

### Build from our forked [ODGI](https://github.com/tonyjie/odgi)
#### No algorithm changes
- Build on [gpu_final](https://github.com/tonyjie/odgi/tree/gpu_final) branch, which is the default branch. You need to have CUDA enabled to build the project. This branch would give you the GPU implementation with all the optimizations excluding warp-level data reuse. Since there is no algorithmic change, the generated layout is the same as the CPU-generated layout. 

#### Explore the max performance with warp-level data reuse optimization
- Build on [gpu_data_reuse](https://github.com/tonyjie/odgi/tree/gpu_data_reuse) branch. You need to have CUDA enbaled to build the project. This branch will give you the GPU implementation with all the optimizations applied. Since warp-level data reuse optimization changes the algorithm, the generated layout could be different compared to the CPU-generated layout. 

## Usage
### GPU Implementation - No algorithm changes
Simply adding `--gpu` is all you need. We also encourage you to use more `<NUM_CPU_THREADS>`, as there is still a preprocessing part that is done by the CPU. They can be parallelized with multiple CPU threads. 

`odgi layout -i <OG_FILE> -o <LAY_FILE> -X <PATH_FILE> --threads <NUM_CPU_THREADS> --gpu`

### GPU Implementation - explore design space with warp-level data reuse
Two tuned arguments are used to explore the design space
- data reuse factor `--data-reuse`
- step reduction factor `--step-decrease` 

`odgi layout -i <OG_FILE> -o <LAY_FILE> -X <PATH_FILE> --threads <NUM_CPU_THREADS> --gpu --data-reuse <DATA_REUSE_FACTOR> --step-decrease <STEP_REDUCTION_FACTOR>`

## Dataset
Human Pangenome Graph [dataset](https://github.com/human-pangenomics/HPP_Year1_Assemblies) from Human Pangenome Reference Consortium ([HPRC](https://humanpangenome.org/)). 

