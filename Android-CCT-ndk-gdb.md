## gdb Native Programs on Android    

从host链接在device（arm设备）上的进程进行gdb调试，两者分别称为host和device。   

### 1. Host端配置   
Windows or Linux is ok，分别列举两个平台上建立工具交叉编译工具链的方法。   

- On Windows   

```
# step 1. Download windows or linux version andriod-ndk

# step 2. Windows
python2.7  make_standalone_toolchain.py  --arch=arm64  --install-dir=/D/AndroidARM  --force

# step 3 修改window环境变量，把路径添加到path中，windows在查找库时不区分lib和bin
change env var: path
```   

- On Linux  
 
```   
# step 1. linux
python2.7  make_standalone_toolchain.py  --arch=arm64  --install-dir=~/luxingjing/project/ASR-decoder/android-ndk-r12b/AndroidARM  --force
# 
export PATH=
export LD_LIBRARY_PATH
```   

### 2. Host端编译程序   
- 编译选项中加入-g -O0   
- 将编译好的程序发送到device端   

### 3. Device端启动gdbserver   
64bit code with gdbserver64.  

```
gdbserver :12345 ./test

# 结果
rk3399_mid:/data/local/asr-speedup # gdbserver64 :12345 ./test
Process ./butterfly-demo created; pid = 7448
Listening on port 12345
Remote debugging from host 127.0.0.1

```

### Host端发起gdb（window系统为例）

- Bind port on Windows host   

```   
adb forward tcp:12345 tcp:12345
```

- 开始 gdb   

```  
gdb.exe test

# 启动gdb后，设定target
target remote :12345

# 设置断点并进行调试，用continue代替run
b main
continue
```  


## 备注：
- host和device的gdb，gdbserver版本一致  
- sysroot路径要加入到环境变量     
- 配置工具链主要选择C++版本，libc++只支持llvm   
