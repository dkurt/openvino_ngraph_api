# OpenVINO nGraph API tutorial project

This project demonstrates basic usage of OpenVINO API to build
Intermediate Representation networks in runtime and then infer it. To load
native networks topologies OpenCV is used. It also provides reference outputs to
compare accuracy.

**NOTE** By default, OpenCV already uses OpenVINO as computational backend (read more [here](https://github.com/opencv/opencv/wiki/Intel%27s-Deep-Learning-Inference-Engine-backend)). So this project might be useful only if you want to emit OpenCV dependency or you have a format of deep learning networks
which is not supported by OpenCV readers yet.

## Getting started

1. Download OpenVINO 2020.1: https://software.seek.intel.com/openvino-toolkit

2. Clone this repository

    ```sh
    git clone https://github.com/dkurt/openvino_ngraph_api
    cd openvino_ngraph_api
    git lfs pull
    ```

3. Setup environment variables

    ```sh
    source /opt/intel/openvino/bin/setupvars.sh
    export ngraph_DIR=/opt/intel/openvino/deployment_tools/ngraph/cmake
    ```

4. Build using CMake

    ```sh
    mkdir build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release && make -j8
    ```

5. Test one of the topologies

    ```sh
    $ ./ngraph_demo alexnet

    l1 diff: 4.3695e-10
    lInf diff: 3.25963e-09
    ```

    ```sh
    $ ./ngraph_demo squeezenet

    l1 diff: 7.49727e-10
    lInf diff: 1.41561e-07
    ```

    ```sh
    $ ./ngraph_demo ssd

    [ OK ] matched class 3 with confidence 0.46
    [ OK ] matched class 7 with confidence 0.99
    [ OK ] matched class 12 with confidence 0.68
    [ OK ] matched class 13 with confidence 1.00
    [ OK ] matched class 15 with confidence 0.96
    [ OK ] matched class 15 with confidence 0.36
    ```
