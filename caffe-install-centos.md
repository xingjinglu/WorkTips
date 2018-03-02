## Install [1]

- 1. Install prerequest
  * Use the default the python is better, such /usr/bin/python
  * boost depend on the python, make sure it use /usr/bin/python
  * protoc is compiler and  protobuf is library, install both and set the path.
  * OpenBLAS, it is ok to install the release version.

- 2. configure
```
  
  1) Add sth in the CMakeLists.txt
     cmake_minimum_required(VERSION 2.8.7)
     SET(BOOST_ROOT /search/speech/luxingjing/software/boost166)
     SET(BOOST_INCLUDEDIR /search/speech/luxingjing/software/boost166/include)
     SET(BOOST_LIBRARYDIR /search/speech/luxingjing/software/boost166/lib)

  2) run cmake
     $cd caffe
     $mkdir -p build
     $cd build
     $cmake -DBLAS=open   ../

  3) Make the protoc version is right in the CMakeCache.txt
     //The Google Protocol Buffers Compiler
     Protobuf_PROTOC_EXECUTABLE:FILEPATH=/usr/local/bin/protoc

```

- 3. make 
- 4. make install 


## Use NCNN

### 1. run AlexNet with NCNN[3]

- 1. error
```
$ upgrade_net_proto_text deploy.prototxt deploy_new.prototxt 
Error parsing text-format caffe.NetParameter: 7:1: Expected identifier, got: <
Failed to parse input text file as NetParameter: deploy.prototxt
```






[1]  http://caffe.berkeleyvision.org/install_yum.html
[2] 
[3] https://github.com/Tencent/ncnn/wiki/ncnn-%E7%BB%84%E4%BB%B6%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8C%97-alexnet

