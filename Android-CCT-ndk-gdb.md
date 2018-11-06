## gdb Native Programs on Android 

## Host 
Windows or Linux is ok.

- Install cross-compile-toolchain 



```
# step 1. Download windows or linux version andriod-ndk

# step 2. Windows
python2.7  make_standalone_toolchain.py  --arch=arm64  --install-dir=/D/AndroidARM  --force
change env var: path

# step. linux
python2.7  make_standalone_toolchain.py  --arch=arm64  --install-dir=~/luxingjing/project/ASR-decoder/android-ndk-r12b/AndroidARM  --force
export PATH=
export LD_LIBRARY_PATH
```

- Bind port
```
adb forward tcp:1234 tcp:1234
```

- Do gdb
```
gdb/gdb.exe test
target remote :1234

continue
```

## Device
64bit code with gdbserver64.  

gdbserver :1234 ./test


## 备注：
- host和device的gdb，gdbserver版本一致  
- sysroot路径要加入到环境变量     
- 配置工具链主要选择C++版本，libc++只支持llvm   





