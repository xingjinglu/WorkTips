#ifndef SINGLETON_H_
#define SINGLETON_H_

#include<iostream>

class Singleton {
 public:
    static Singleton* GetInstance() {
        return instance_;
    }

 private:
    Singleton() {std::cout<<"hello\n";}
    static Singleton* instance_;
};

#endif  // SINGLETON_H_
