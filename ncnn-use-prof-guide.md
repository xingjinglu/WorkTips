[TOC]

## ncnn [1]

### 1. build: [1]   
  It works as the instruction in [1].
  Make sure the version android-ndk is r15c.

  ```
  $cmake -DCMAKE_TOOLCHAIN_FILE=/search/speech/luxingjing/software/android-ndk-r15c/build/cmake/android.toolchain.cmake  -DANDROID_ABI="arm64-v8a" -DANDROID_ARM_NEON=ON     -DANDROID_PLATFORM=android-14 ..

  $make

  $make install

  ```

- build tools

  - protobuf：版本太新不行，例如3.1.1,目前测试可以的版本的是2.6.1

  - 需要修改tools/caffe/CMakeLists.txt 和/tools/onnx/CMakeLists.txt文件，添加对应的protobuf路径   

    ```bash
    # caffe/CMakeLists.txt
    set(Protobuf_INCLUDE_DIR "/usr/local/protobuf261/include")
    set(Protobuf_LIBRARY "/usr/local/protobuf261/lib")
    link_directories("/usr/local/protobuf261/lib")
    
    target_link_libraries(caffe2ncnn  PRIVATE protobuf)
    # onnx/CMakeLists.txt 
    添加相同配置信息
    ```

    

### 2. Run Squeezenet with ncnn [6] 

Uncomment add_subdirectory(examples)  in CMakeList.txt of top directory.   

- 1. Install Opencv with CUDA9.0 on CentOS [5]
bugs: CMakeList.txt not fit cuda9.0, fix these errors accord to [4].   

Don't need to cross-compile the OpenCV libs.   

```
// Step 1. Install OpenCV

// Step 2. Setup enviroment. 
// Setup enviroment vars of OpenCV.
sudo gedit /etc/ld.so.conf.d/opencv.conf  
// Insert  /usr/local/lib
sudo ldconfig


sudo gedit /etc/bash.bashrc 
// Insert the follow 2 lines:
PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/lib/pkgconfig
export PKG_CONFIG_PATH
sudo source /etc/bash.bashrc

//  Step 3. Check opencv:
$python
$import cv2

```
It works. 

Add belows code in CMakeList.txt   
```
set(OpenCV_DIR /search/speech/luxingjing/software/ncnn/third_party/OpenCV-android-sdk/sdk/native/jni/abi-arm64-v8a) 
```

But, it is sad on build tool.



### 3. build benchmark with ncnn

```
1) Get the OpenCV libs  
mkdir third_party  
cd third_party  
wget https://github.com/opencv/opencv/releases/download/3.4.0/opencv-3.4.0-android-sdk.zip
unzip opencv-3.4.0-android-sdk.zip

2) Modify the CMakeList.txt
set(OpenCV_DIR /search/speech/luxingjing/software/ncnn/third_party/OpenCV-android-sdk/sdk/native/libs/arm64-v8a)

3) Build
cmake -DCMAKE_TOOLCHAIN_FILE=/search/speech/luxingjing/software/android-ndk-r15c/build/cmake/android.toolchain.cmake  -DANDROID_ABI="arm64-v8a" -DANDROID_ARM_NEON=ON  -DNCNN_OPENCV=ON -DNCNN_BENCHMARK=ON    -DANDROID_PLATFORM=android-14 ..

4) make

5) copy benchncnn and params under ~/software/ncnn/benchmark  to the Android-Soc
adb push .\benchncnn /data/local/test/ncnn_benchmark/
adb push .\alexnet.param .\googlenet.param .\mobilenet.param .\mobilenet_ssd.param .\mobi
lenet_v2.param .\resnet18.param .\shufflenet.param .\squeezenet.param .\squeezenet_ssd.param .\vgg16.param /data/local/t
est/ncnn_benchmark

#./benchncnn 8 4 0

./benchncnn [loop count] [num threads] [powersave]


```
 It works.

 -DNCNN_BENCHMARK=ON :  the options of profiling network layers.

### 4. build tools 
It failed now.  



### Reference
[1] https://github.com/Tencent/ncnn/wiki/how-to-build    
[2] https://github.com/Tencent/ncnn/wiki/how-to-use-ncnn-with-alexnet  
[3] https://github.com/BUG1989/ncnn-benchmark 
[4] https://stackoverflow.com/questions/46584000/cmake-error-variables-are-set-to-notfound   
[5] http://blog.csdn.net/u011557212/article/details/54706966?utm_source=itdadao&utm_medium=referral  
[6] http://blog.csdn.net/fuwenyan/article/details/76576708   

### 5. Performance evalueate on RK3399 
   Setup at the peak performance for each core.
   A53: 1.415GHz
   A72: 1.8GHz

- 1. benchncnn 8 1 0  @ A72      


|    squeezenet  |min =  176.65 |max =  186.99 |avg =  179.97|
|--|--|--|--|
|      mobilenet |min =  297.26 |max =  319.95 |avg =  311.13|
|   mobilenet_v2 |min =  367.38 |max =  401.66 |avg =  382.13|
|     shufflenet |min =   99.13 |max =  102.18 |avg =   99.99|
|      googlenet |min =  571.57 |max =  596.20 |avg =  586.63|
|       resnet18 |min =  888.14 |max = 1059.80 |avg =  929.63|
|        alexnet |min =  529.83 |max =  541.63 |avg =  532.81|
|          vgg16 |min = 2594.72 |max = 2961.71 |avg = 2805.33|
| squeezenet-ssd |min =  274.25 |max =  321.47 |avg =  306.92|
|  mobilenet-ssd |min =  253.50 |max =  317.98 |avg =  291.48|



- 2. benchncnn 8 2 0  @ A72   

|     squeezenet |min =   80.69 |max =  118.43 |avg =   97.37|
|--|--|--|--|
|      mobilenet |min =  166.73 |max =  245.18 |avg =  200.62|
|   mobilenet_v2 |min =  174.31 |max =  196.35 |avg =  180.34|
|     shufflenet |min =   50.90 |max =   51.61 |avg =   51.13|
|      googlenet |min =  292.92 |max =  311.28 |avg =  297.57|
|       resnet18 |min =  354.39 |max =  391.45 |avg =  368.41|
|        alexnet |min =  262.68 |max =  286.97 |avg =  274.74|
|          vgg16 |min = 1631.17 |max = 1874.95 |avg = 1700.03|
| squeezenet-ssd |min =  137.27 |max =  224.79 |avg =  155.05|
|  mobilenet-ssd |min =  165.53 |max =  195.96 |avg =  179.02|


- 3. benchncnn 8 4 0   

|     squeezenet |min =  153.16 |max =  170.70 |avg =  157.52|
|--|--|--|--|
|      mobilenet |min =  170.58 |max =  389.48 |avg =  257.77|
|      shufflenet| min =   97.06| max =  305.98|avg =  123.94|
|      googlenet |min =  302.39 |max =  523.80 |avg =  335.98|
|       resnet18 |min =  362.40 |max =  535.84 |avg =  401.67|
|        alexnet |min =  266.49 |max =  440.86 |avg =  297.67|
|          vgg16 |min = 1716.23 |max = 1938.85 |avg = 1871.01|
| squeezenet-ssd |min =  167.41 |max =  427.96 |avg =  227.11|
|  mobilenet-ssd |min =  177.80 |max =  450.80 |avg =  227.79|

- 4. ./benchncnn 8 1 0 @ A53  

|    squeezenet | min =  389.93 |max =  400.36 |avg =  395.01|
|--|--|--|--|
|      mobilenet| min =  594.31 |max =  609.08 |avg =  598.93|
|   mobilenet_v2| min =  636.34 |max =  646.07 |avg =  642.80|
|     shufflenet| min =  212.10 |max =  214.89 |avg =  213.14|
|      googlenet| min = 1264.61 |max = 1277.45 |avg = 1269.55|
|       resnet18| min = 2114.45 |max = 2129.91 |avg = 2122.99|
|        alexnet| min = 1582.26 |max = 1816.82 |avg = 1726.44|
|          vgg16| min = 8257.97 |max = 8344.18 |avg = 8299.59|
| squeezenet-ssd| min =  821.89 |max =  881.00 |avg =  858.53|



- 5. ./benchncnn 8 2 0 @ A53  

|     squeezenet |min =  232.41 |max =  243.20 |avg =  237.88|
|--|--|--|--|
|   mobilenet_v2 |min =  242.76 |max =  437.25 |avg =  394.59|
|     shufflenet |min =   86.16 |max =   88.54 |avg =   87.02|
|      googlenet |min =  710.59 |max =  718.99 |avg =  715.46|
|       resnet18 |min = 1152.49 |max = 1168.78 |avg = 1158.13|
|        alexnet |min =  828.41 |max =  845.38 |avg =  833.22|
|          vgg16 |min = 3668.70 |max = 4675.28 |avg = 4143.42|
| squeezenet-ssd |min =  490.62 |max =  499.39 |avg =  495.11|




- 6. ./benchncnn 8 4 0 @ A53  

|squeezenet|  min =  128.97| max =  139.76|   avg =  132.71|
|--|--|--|--|
|  mobilenet| min =  193.60|  max =  235.33|  avg =  207.23|
| mobilenet_v2|min =  178.05|  max =  218.81  | avg =  196.90|
|   shufflenet| min =   72.84|  max =   76.11 | avg =   73.94|
|    googlenet|  min =  354.38|  max =  387.24| avg =  363.56|
|     resnet18|  min =  487.47|  max =  548.42| avg =  513.95|
|      alexnet|  min =  392.68|  max =  419.04| avg =  398.83|
|        vgg16|  min = 1966.30|  max = 2145.15| avg = 2038.04|
|squeezenet-ssd|  min =  190.68|  max =  208.47| avg =  200.53|
| mobilenet-ssd|  min =  206.74|  max =  241.38| avg =  225.03|










>>>>>>> 37fd569d54de71769cd779a45fe7519aefcddc3e
