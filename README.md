# Drone-race

## Installation

This project requires a Nvidia GPU running Cuda >= 12.4, and allows for Python rendering with matplotlib. Possible usages are:
- running the Python rendering part on your local machine, and running the C++/Cuda backend over SSH on a machine with a strong GPU (see Setting up a remote connection below)
- running everything over SSH, with only CLI output (or matplotlib through X11 forwarding, but interaction will be very slow)
- running everything locally if your machine is CUDA-capable

In the first case, you will need to run the installation commands twice (on your machine and over SSH). This will install cuda toolkit on your machine (and Python libraries on the SSH machine) which will not be used; you can remove them in environment.yml if this is an issue.

To install everything:

1. Install Conda (miniconda works fine)
2. git clone the repo
3. `conda env create -f environment.yml`. This will create a conda environment named `drone-race` (you can change its name in `environment.yml`).

## Running the project

### C++ side

Activate the environment: `conda activate drone-race`.
Use the provided CMake configuration to compile:

```
mkdir build && cd build
cmake ..
make -j8
```

If CMake fails to compile Cuda code (with nvcc outputting an error like `cannot find "cuda_runtime.h"`), it might be because Conda installed CUDA headers in `$CONDA_PREFIX/targets/x86_64-linux/include/` instead of `$CONDA_PREFIX/include`. Check if the former contains `cuda_runtime.h`, and if so, create symbolic links to make sure nvcc is able to find its own headers:
```
ln -s $CONDA_PREFIX/targets/x86_64-linux/include/* $CONDA_PREFIX/include/
ln -s $CONDA_PREFIX/targets/x86_64-linux/lib/* $CONDA_PREFIX/lib/
```

Then, run `./build/drone_race`. It will keep listening for configurations sent by the Python side until you ctrl-C (this avoids re-running the engine for each new simulation).

If you get an error `zmq::error_t: address already in use`, that means the previous run did not close correctly. Run `killall drone_race` to properly close the sockets.

### Python side

Activate the environment: `conda activate drone-race`.
Then, simply run `python python-src/main.py`. It will send the given configuration to the backend, listen for the results, and display them live.

### Setting up a remote connection

If the C++ backend is running on a remote machine, use the following command in a new terminal to set up port forwarding:

```
ssh -L 5555:localhost:5555 -q user@remote
```

You can just leave it running in the background (you might need to restart it if your machine wakes up from sleep mode).

### VSCode usage

To ease up development, make sure that you correctly select the conda interpreter on the Python side (running `Python: Select Interpreter` if necessary).

On the C++ side, to get correct Intellisense behavior, you can use the following `.vscode/c_cpp_properties.json`:
```json
{
    "configurations": [
        {
            "name": "CUDA-Conda",
            "includePath": [
                "${workspaceFolder}/**",
                "${env:CONDA_PREFIX}/include"
            ],
            "compilerPath": "/usr/bin/g++",
            "cStandard": "c17",
            "cppStandard": "c++20",
            "intelliSenseMode": "linux-gcc-x64",
            "compileCommands": "${workspaceFolder}/build/compile_commands.json"
        }
    ],
    "version": 4
}
```

To make sure the CMake extension works well, add this to your `.vscode/settings.json`:

```json
    "cmake.configureSettings": {
        "CMAKE_PREFIX_PATH": "${env:HOME}/miniconda3/envs/drone-race",
        "CMAKE_CUDA_HOST_COMPILER": "${env:HOME}/miniconda3/envs/drone-race/bin/x86_64-conda-linux-gnu-g++"
    },
    "cmake.environment": {
        "CONDA_PREFIX": "${env:HOME}/miniconda3/envs/drone-race",
        "PATH": "${env:HOME}/miniconda3/envs/drone-race/bin:${env:PATH}",
        "LD_LIBRARY_PATH": "${env:HOME}/miniconda3/envs/drone-race/lib:${env:LD_LIBRARY_PATH}"
    }
```

(modifying `/miniconda3/envs/drone-race` by the value of `$CONDA_PREFIX` in your environment if it does not match).
