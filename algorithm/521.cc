#include<string>
#include<iostream>

int getNum(int n)
{
  int fiveNum = n / 5;
  int remain = n % 5;
  int totalNum = 0;
  for(int i = 0; i <= fiveNum; i++)
    totalNum += (n  - i * 5) /  2 + 1;

  return totalNum;
}


// n = 5x + 2y + z;
// n = 10,  2 0 0; 1 2 1; 1 1 3; 1 0 5; 0 5 0; 0 4 2; 0 3 4; 0 2 6; 0 1 8; 0 0
// 10
int main(int argc, char **argv)
{
  std::cout<<"argc = " << argc << std::endl;
  if(argc < 2){
    std::cout<<"should have a int numer as input \n";
    return 0;
  }

  int n = std::stoi(argv[1]);
  
  int total = getNum(n);
  std::cout<<"total = " << total << std::endl;
  return 0;
}
