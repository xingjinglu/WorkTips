# Methods to profile application on Android+ARM systems.   
## Many methods as below:
### 1. [DS-5 Development Studio](https://developer.arm.com/products/software-development-tools/ds-5-development-studio/streamline)  
  Its function like VTune of intel, a good tool to analyze the    
  performance bottleneck of applican and systems.   

### 2. android_ndk_perf @ Google: [fplutil](https://google.github.io/fplutil/android_ndk_perf.html)   
It is said the tool only support two devices, and I don't try it .

### 3. [Tegra System Profiler @ NVIDIA](https://developer.nvidia.com/tegra-system-profiler)   
Its functions seems like DS-5.   

### 4. [android-ndk-profiler](https://github.com/richq/android-ndk-profiler)
The same tool as "gprof" for linux + x86 systems. There are user  
 docs, and I figured out how to use the tool on ARM A72 + Android.

### 5. simpleprof of android-ndk-r14b[1]  
 It is the same tool like "perf" for linux, and it can collect info about cpu cycles, events and    
 program execution time.   


## User guide of android-ndk-profiler   // light weight
### 1. git clone the tool  
```
 $git clone https://github.com/richq/android-ndk-profiler  
```
### 2. build the static library  
  ```  
  $cd android-ndk-profiler/jni  
  $ndk-build  
  ```  
- Tips:   
  + If need "pic" library, you should add -Wl,-pic in the    Android.mk
  + It does not support arm64-v8a.  

### 3. Setup your project, such as: hellojni  
       Modify your Android.mk.  
#### 3.1  import module  
```  
APP_DEBUG := $(strip $(NDK_DEBUG))  
ifeq ($(APP_DEBUG),1)  
  LOCAL_CPPFLAGS += -pg  
  LOCAL_STATIC_LIBRARIES:=android-ndk-profiler  
endif  
```  
#### 3.2 Build your *.so files  
```  
include $(CLEAR_VARS)  
LOCAL_MODULE := gemmBatch  
LOCAL_SRC_FILES += gemm_batch.cpp  
LOCAL_CPPFLAGS += -O3 -Ofast  -g  -fexceptions -frtti -std=c++11 -Wl,-pic  
LOCAL_CPPFLAGS += -Winline  
LOCAL_CPPFLAGS += -mfpu=neon -mfloat-abi=softfp  
LOCAL_CPPFLAGS += -v -I   /search/speech/xingjing/software/android-ndk-profiler/jni   

LOCAL_LDLIBS = -Wl,-rpath /search/odin/xingjing/software/AndroidARM/sysroot/usr/lib  -llog   
        include $(BUILD_SHARED_LIBRARY) 
``` 
- Tips
  + Add the source file path of android-ndk-profiler
  ```
    -I /search/speech/xingjing/software/android-ndk-profiler/jni
  ```
  + Add flags: -g    

#### 3.3 Change your code to specify the code-scope to be profiled      
    I.E. in the main() funciton I insert the following codes:    
```  
    monstartup("libgemmBatch.so");  // the .so build above.  
    .....  
    // code do real computation.  
    .....  
    setenv("CPUPROFILE", "/data/local/test/gmon.out", 1); // 事件  
    moncleanup();  
```    
- Tips:
  + Other controls:  
    ```
    setenv("CPUPROFILE_FREQUENCY", "500", 1);  // 采样频率   
    ```
  + Output gmon.out to some dir, like /data/local/test/  
  
#### 3.4 Get the gmon.out and do analysis  
```
$adb pull /data/local/test/gmon.out  
$aarch64-linux-android-gprof   hellojni/obj/local/armeabi-v7a/libgemmBatch.so  
```
- Tips:  
  + ndk-build会生成两种版本的.so， 有符号的版本在obj目录下，  
    另一个版本在libs目录下，在分析数据时，用有符号的版本。  
    

### 4. 实验结果如下  
```  
Flat profile:    

Each sample counts as 0.00333333 seconds.  

  %   | cumulative|  self |     | self  | total |             |        
 time |    seconds|seconds|calls|Ts/call|Ts/call|name         |  
 -----|-----------|-------|-----|-------|-------|-------------|      
 28.40|      0.31 |   0.31|     |       |       |gemmBatchBase|    
 24.38|      0.57 |   0.26|     |       |       |gemmBatchOpt2|     
 24.38|      0.83 |  0.26 |     |       |       |gemmBatchOpt3|      
 22.53|      1.08 |  0.24 |     |       |       |gemmBatchOpt1|     

 %         the percentage of the total running time of the  
time       program used by this function.  

cumulative a running sum of the number of seconds accounted  
 seconds   for by this function and those listed above it.  

 self      the number of seconds accounted for by this  
seconds    function alone.  This is the major sort for this  
           listing.  

calls      the number of times this function was invoked, if  
           this function is profiled, else blank.   
 
 self      the average number of milliseconds spent in this  
ms/call    function per call, if this function is profiled,  
	   else blank.  

 total     the average number of milliseconds spent in this  
ms/call    function and its descendents per call, if this   
	   function is profiled, else blank.  

name       the name of the function.  This is the minor sort  
           for this listing. The index shows the location of  
	   the function in the gprof listing. If the index is  
	   in parenthesis it shows where it would appear in  
	   the gprof listing if it were to be printed.  
```


[1] https://android.googlesource.com/platform/system/extras/+/master/simpleperf/ 
