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

5. Test one of the topologies (model files can be downloaded by URLs from https://github.com/opencv/opencv_extra/blob/4.x/testdata/dnn/download_models.py)

    ```sh
    $ ./ngraph_demo alexnet

    Reference range: [1.05861e-07, 0.655111]
    l1 diff: 4.21329e-10
    lInf diff: 8.9407e-08
    inference time: 17.4656ms
    ```

    ```sh
    $ ./ngraph_demo squeezenet

    Reference range: [2.68397e-08, 0.258784]
    l1 diff: 1.21957e-09
    lInf diff: 2.38419e-07
    inference time: 6.62212ms
    ```

    ```sh
    $ ./ngraph_demo ssd

    [ OK ] matched class 3 with confidence 0.46
    [ OK ] matched class 7 with confidence 0.99
    [ OK ] matched class 12 with confidence 0.68
    [ OK ] matched class 13 with confidence 1.00
    [ OK ] matched class 15 with confidence 0.96
    [ OK ] matched class 15 with confidence 0.36
    inference time: 20.7912ms
    ```

    ```sh
    $ ./ngraph_demo alexnet_quant

    Reference range: [1.26981e-07, 0.625701]
    l1 diff: 6.70524e-05
    lInf diff: 0.0294101
    inference time: 11.1459ms
    ```

Tested on `Intel(R) Core(TM) i5-4460  CPU @ 3.20GHz x4` with Intel OpenVINO 2020.2
