###



### Build and run opencl examples  

- 1. install scons  

```
outnet wget https://sourceforge.net/projects/scons/files/scons/3.0.1/scons-3.0.1.tar.gz   
tar xzf scons-3.0.1.tar.gz   
cd scons-3.0.1/   
python setup.py install 
```

- 2. build library for Android[1]   

```
CXX=clang++ CC=clang scons Werror=1 -j8 debug=0 asserts=1 neon=0 opencl=1 embed_kernels=1 os=android arch=arm64-v8a

// Build libOpenCL.so
aarch64-linux-android-gcc -o libOpenCL.so -Iinclude -shared opencl-1.2-stubs/opencl_stubs.c -fPIC -shared 

//  -L build/   the static library
aarch64-linux-android-clang++ examples/cl_sgemm.cpp utils/Utils.cpp -I. -Iinclude -std=c++11 -L build/ -larm_compute-static -larm_compute_core-static -L. -o cl_sgemm -static-libstdc++ -pie -lOpenCL -DARM_COMPUTE_CL

```

- 3. Run the cl_sgemm on Anroid[2]    
```
ln -s /system/lib64/egl/libGLES_mali.so /data/local/tmp/libOpenCL.so  
LD_LIBRARY_PATH=. ./cl_convolution_aarch64  
```

### GDB Android ACL apps with Android-Strudio   
[3,4,5,6]   




### Refernce  
[1] https://arm-software.github.io/ComputeLibrary/v17.10/index.xhtml#S3_3_android   
[2] https://github.com/ARM-software/ComputeLibrary/issues/111  
[3] https://github.com/ARM-software/ComputeLibrary/issues/52  
[4] https://github.com/ARM-software/ComputeLibrary/issues/137  
[5] https://github.com/ARM-software/ComputeLibrary/issues/98  
[6] https://github.com/ARM-software/ComputeLibrary/issues/135  
