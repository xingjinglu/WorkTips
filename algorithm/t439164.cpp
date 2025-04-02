#include <iostream>
#include <string>
using namespace std;

// a = 123456789123456789123456789
// b = 234567890234567890234567890
//  1234 -  123, 
//  123  - 12344, 
//  124 - 256
int main()
{
  // a - b
  std::string a, b;
  std::cin >>a >> b;

  int alen = a.length() - 1;
  int blen = b.length() - 1;

  int i = alen;
  int j = blen;

  std::string result = "";
  // "23" + "456" = "23456"
  int jiewei = 0;

  // 位数相同部分，逐步做减法 
  // 1458 - 245,   124 - 1758,  245 - 456,
  while(i >= 0 && j >= 0){
    int a1 = a[i] - '0';
    int b1 = b[j] - '0';
    int c = a1 - b1 - jiewei;
    if(c < 0  ){
      jiewei = 1;
      if(i != 0)
        c = 10 + c; // c=-2 ==>  jiewe=1， c=8
      else{
        c = 0 - c;
        std::cout<<"i == 0" << endl;
      }
    }
    else
      jiewei = 0;


    result = std::to_string(c) + result; // '8' + '2' ==> "82"
    i--;
    j--;

  }

  // 如果a的长度大于b
  // 156 7 234 - 123
  if(i >= 0){
    while(i >= 0){
      int a1 = a[i] - '0';
      int c = a1 - 0 - jiewei;
      if(c < 0){
        jiewei = 1;
        c = 10 + c;
      }else
        jiewei = 0;
      result = std::to_string(c) + result; // '8' + '2' ==> "82"
      i--;
    }

  }
  // 如果a的长度小于b
  else if(j >= 0){
    while(j >= 0){
      int b1 = b[j] - '0';
      int c = 0 - b1 - jiewei;
      if( c < 0 ){
        jiewei = 1;
        c = 0 - c;
      }else
        jiewei = 0;

      result = std::to_string(c) + result; // '8' + '2' ==> "82"
      j--;
    }
  }
  // 如果a的长度等于b, 不需要特殊处理
  
  if(jiewei)
    result = '-' + result;
  //

  std::cout<< result << std::endl;

  return 0;
}
