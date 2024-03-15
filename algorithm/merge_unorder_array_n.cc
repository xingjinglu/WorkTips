#include<iostream>
#include<vector>

struct node{
  int val_;
  struct node *left_;
  struct node *right_;
}

class Heap{
  public:
    node(int max_num){ 
      max_num_ = max_num; 
      cur_num_ = 0;
    }

    int max_num_; 
    int cur_num_; 
    int val_;
    class node *root_;
    class node *left_;
    class node *right_;
};

// max_heap
int heap_sort(node &hp, int val)
{

  // Adjust heap.
  if (hp.cur_num_ >= hp.max_num_ ){

  }
  // insert new node.
  else{
    nd = new node
  }
  
  return 0;
}

int impl_naive()
{
  std::vector<int> a{1, 7, 8, 20, 3, 4};
  std::vector<int> b{22, 9, 7, 5, 4, 3};
  int N = 4;


  std::cout<<N<< std::endl;

  return 0;
}

int main()
{
  impl_naive();

  return 0;
}
