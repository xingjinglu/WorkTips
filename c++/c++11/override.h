#pragma once


class Rect{
  public:
    Rect(){}
    ~Rect(){}
    Rect(float h, float w) : h_(h), w_(w){
    }
   virtual float getArea(){return h_ * w_;}
  public:
    float h_;
    float w_;
};


class Triangle : public Rect{
  public:
    Triangle(){}
    Triangle(float h, float w){
      h_ = h;
      w_ = w;
    }
    ~Triangle(){}
    float getArea(){return (h_ * w_)  / 2;}
 // private:
 //   float h_;
 //   float w_;
};

class Circle {
  public:
    Circle(double r) : r_(r){}
  public:
    double r_;
    static double pi_;
    double getArea(){return pi_ * r_ * r_;}

};
