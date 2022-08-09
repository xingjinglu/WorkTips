#pragma once
#include<iostream>


class Base{
  public:
    Base(){}
    Base(int x, int y):x_(x), y_(y){}
    ~Base(){}
    void print(){std::cout<<"x_, y_: " << x_ <<", " << y_ << std::endl;}

    friend class Cfriend;

  private:
    int x_;
    int y_;
};



class Cfriend{
  public:
   void test(Base &b){b.x_ = 10;}
};

class Derived : public Base{

  public:

  private:

};
