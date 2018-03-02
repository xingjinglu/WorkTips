# ARM, C/C++/OpenCL对float16的支持

## ARM平台支持的数据类型

### 整型数据：byte, halfword, word, double word, quad-word
### 浮点数据：
  - half：ARM支持两种，IEEE754标准以及alternative格式，半精度数据只能与float进行相互转换，对于arm32, arm64皆是。
    - IEEE754：当指数为31时，会返回inf或者nan
    - alternative：当指数为31时，进行正规化：-1S × 216 × (1.fraction)  
  - float
  - double    

### 其他数据  

  - fixed point interpretation of words, double words
    - 浮点数据类型
    - 定义了小数点的位置
    - 向量数据类型
    - polynomial：值为0或者1，其上的加法类似于or，乘法与标量乘法类似。  
    
### Tips

- 文档[1]是NVIDIA关于half数据类型的定义，比较明确，应该是符合IEEE754标准。
- ARM的手册中给出半精度的计算方法，当指数为31时，采用了一个替代算法，而不是给出inf或者nan;  
- 另外ARM平台对于指数为31的情况定义了alternative处理方法，具体参考[2]。   

## ARM对half float的支持说明

- pre - ARMv8.1  
  - ARM在C的扩展中增加了__f16关键字，该关键字不能用于数值计算、不能用于参数或者返回值。在数值计算过程中，该类型至少被提升为float，就像char，short在计算过程中被提升为int来计算，实际硬件中没有支持char,fp16或者short。   
  - __fp16 *可以用作参数或者返回值；

- 所有ARMv8架构
  - ARM支持两种格式半精度浮点数据类型的表示分别是IEEE754-2008标准和ARM alternative，用户可以用下面的宏定义来选择对应的版本。
__ARM_FP16_FORMAT_IEEE  
__ARM_FP16_FORMAT_ALTERNATIVE  
  - 默认是IEEE标准，建议在实际应用程序中采用该标准。
  - __ARM_FP16_ARGS  =1， 表示支持__fp16可以用作参数或者返回值  
  - __ARM_FEATURE_FP16_SCALAR_ARITHMETIC 表示支持标量的half数值运算  
  - __ARM_FEATURE_FP16_VECTOR_ARITHMETIC …向量的数值运算。

## C/C++程序中的半精度浮点类型

- ACLE中扩展了__fp16  
  - __fp16是C/C++针对AARCH64做的扩展  
  - typedef __fp16 float16_t; // arm_neon.h  
     所以，在arm平台上也可以用float16_t描述半精度  

- ISO/IEC中扩展了_Float16
  - gcc已经支持_Float16;但是目前用到的clang3.8交叉编译器还不支持。  

## 双精度/单精到半精的类型转换

### double2half  

从ARMv8.1开始支持double到half的转换。   
```
double xd;
__fp16 xs = (float)xd;
```

编译时候选项：-mfp16-format -mfp16-format=ieee -mfp16-format=alternative  

- Tips
  - 当硬件平台支持类型转换时，调用相应的指令；否则通过库来实现。

### float2half & half2float
深度学习领域，经常用到float和half之间的数据类型转换，下面给出具体的实现细节。
- float2half
  - OpenCL平台  
OpenCL提供了float转换half的intrinsic接口
  - C++  
有一个开源C++库实现了float和half之间的数据类型转换,half.hpp
  - ARM  
提供了NEON instrinsic实现float到half之间的数据类型转换
  - x86  
提供了相应的Intrinsic接口实现，两者之间的互相转换

  - 具体算法,参考文档[4]中的介绍。  
    - Tips：
      - ARM：整数或定点数到浮点数是向最接近的数靠近；  
             浮点数到整数或者定点数是使用向零舍入  
      - NEON支持：浮点和整型，浮点和定点，浮点和half之间的转换[3]。  

- half2float
  - 具体算法[4]：  
    - 正常数据  
    - 拷贝符号位；  
    - 拷贝指数位，但是要调整指数为： he-15+127  
    - 尾部：直接拷贝，并在后面加13个0  
    - +/- 0， inf仍然inf；nan仍然为nan；但是subnormal，在float里就是合理的数据了；  

###	ARMv8-A演进对half-float的支持变化

- ARMv8.0-A和ARMv8.1-A[6]  
只能用于存储和转换，不能支持计算。另外，ARM自身没有推出支持ARMv8.1-A的处理器，高通在2017年11月推出一款处理器。   

- ARMv8.2-A[6]
  - 支持对half float数据处理的支持，同时支持AArch32和AArch64的执行状态，并且支持sclalar和NEON的浮点操作。
  - 与对float提供同样的数据操作支持。  


## 关于half float的扩展材料

### Mali GPU对half float的支持
- 支持fp32, fp16, lowp is fp16
- mali gpu的数值计算支持f16。（TODO：再进一步确认）  

### OpenCL
- 支持half float数据类型；在OpenCL应用中为cl_half。  
- half只能用于生成指向buffer的指针
- 只能用vstore_half将float数据转成half；另外也只能用vload_half读取half数据，在该过程中，half数据会被转成float数据。  

### C/C++

- C和C++中，half float被提升为float，然后再进行数值计算。
- ISO/IEC TS 18661-3:2015
  - 增加了新的float数据类型，GNU C、C++开始支持的新float数据类型，_Floatn 
  - 在ARM AArch64平台，默认支持_Float16数据类型。当开发具有可移植性代码时，采用_Float16数据类型。





## 参考资料  
[1] http://docs.nvidia.com/deeplearning/sdk/pdf/Training-Mixed-Precision-User-Guide.pdf   
[2] ARM手册 A1.4.2  
[3] ARM手册 F8.1.57   
[4] fasthalffloatconversion 2008  
[5] https://gcc.gnu.org/onlinedocs/gcc/Half-Precision.html  
[6] https://community.arm.com/processors/b/blog/posts/armv8-a-architecture-evolution  
[7] https://developer.arm.com/products/software-development-tools/compilers/arm-compiler-5/docs/101028/latest/3-c-language-extensions#ssec-fp16-type  
[8] https://gcc.gnu.org/onlinedocs/gcc/Half-Precision.html#Half-Precision   