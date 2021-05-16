#ifndef __INIT_MEM_VAR_H
class Conv{
  public:
    Conv();
    ~Conv();

    int kernelh;
    int kernelw;
    int stridex;
    int stridey;
};

class ConvCpp11{
  public:
    ConvCpp11();
    ~ConvCpp11();

    int kernelh;
    int kernelw;
    int stridex = 1;
    int stridey = 1;
    
    //static int depthwise = 2;  // Fobiiden in c++11.
    const static int depthwise = 2;
   
};

#endif
