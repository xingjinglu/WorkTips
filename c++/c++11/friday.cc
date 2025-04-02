#include <iostream>

int days_month(int year)
{
  int days = 0;

  if(year / 100 == 0 && year % 4 == 0)
    days = 366;
  else if(year % 4 == 0)
    days = 366;
  else
    days = 365;

  return days;
}

int main()
{
  int num = 20;
  int y = 2008;
  int m = 8;
  int d = 8;
  int days = 0;
  int remain_days = 0;

  for(int i = 0; i <  20; i++){
    days = days_month(i + 2009); 
    remain_days += days;

    int remain = remain_days % 7;
    std::cout<<"year " << (i + 2009) << " days:" << days <<",remain_days: " <<
      remain_days<< "remain: " << remain << std::endl;
    if(remain == 0)
      std::cout<<" " << i + 2009<< std::endl;
  }


  return 0;
}
