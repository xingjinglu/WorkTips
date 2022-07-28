#pragma once


class Rect{
  public:
    Rect(){}
    ~Rect(){}
    Rect(float h, float w) : h_(h), w_(w){
    }
    float getArea(){return h_ * w_;}
  private:
    float h_;
    float w_;
};


class Triangle : public Rect{
  public:
    Triangle(){}
    Triangle(float h, float w):h_(h), w_(w){}
    ~Triangle(){}
    float getArea(){return (h_ * w_)  / 2;}
  private:
    float h_;
    float w_;
};
