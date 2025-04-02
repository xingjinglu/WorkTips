#include <iostream>

using namespace std;
int add2()
{
  int n;
  long long sum = 0;
  cin >> n;
  do{
    long long i = 1, x = 1;
    do{
      sum += x;
      x *=++i;
    }while(i <= n);
  }while(--n);


  cout << sum;
  return 0;
}


int add()
{
  int n;
  long long sum = 0;
  cin >> n;
  do{
    long long i = 1, x = 1;
    do{
      sum += (x *= i++);
    }while(i <= n);
  }while(--n);


  cout << sum;
  return 0;
}

int main()
{
  add();

  return 0;
}
