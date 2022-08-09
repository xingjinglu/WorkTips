#include <iostream>
#include "singleton.h"

int main() {

    std::cout<<"begin \n";
    Singleton *s1 = Singleton::GetInstance();
    Singleton *s2 = Singleton::GetInstance();

    std::cout<<"begin2 \n";
    std::cout << "s1地址: " << s1 << std::endl;
    std::cout << "s2地址: " << s2 << std::endl;
    return 0;
}
