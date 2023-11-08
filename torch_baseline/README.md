# PyTorch implementation of Pangenome Graph Layout

### Prerequisites
- torch
- numpy
- matplotlib
- imageio

### Usage
`python torch_graph_layout.py <input> --batch_size <BATCH_SIZE> --cuda`

Supported input includes: 
- DRB1-3123
- mhc

### Example
`python torch_graph_layout.py DRB1-3123 --batch_size 10000 --cuda`

Other arguments: 
- `--draw` will generate the output layout figure after each iteration, and generate a gif combining these figures. 
- `--lay` will generate an ODGI `.lay` layout file. (But it needs other prerequisites.) Then you can use `ODGI` command to generate the 2D visualization. 

### Requirement for using `--lay`
Generating ODGI `.lay` layout file requires ODGI C++ functions and corresponding pybind implementation. Therefore, it requires building the ODGI project. 

- Git clone our [ODGI](https://github.com/tonyjie/odgi/tree/master) repo, checkout this [branch](https://github.com/tonyjie/odgi/tree/zhang_research_extended) and build the project. 
- Then a dynamic library can be created under the `<your_odgi_path>/lib`. In practice, you might also need to load `libjemalloc.so`. The commands as follows should work: 
- `env LD_PRELOAD=<your_path_for_libjemalloc.so> PYTHONPATH=<your_odgi_path>/lib python torch_graph_layout.py <arguments>`