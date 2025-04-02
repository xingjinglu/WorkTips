#include <iostream>
#include <iomanip>
#include <ctime>

int main()
{
    auto t = std::time(nullptr);
    std::cout << t << std::endl;
    t = std::time(nullptr);
    std::cout << t << std::endl;
    t = std::time(nullptr);
    std::cout << t << std::endl;
    t = std::time(nullptr);
    std::cout << t << std::endl;
    auto tm = *std::localtime(&t);
    std::cout << std::put_time(&tm, "%d-%m-%Y %H-%M-%S") << std::endl;
}
