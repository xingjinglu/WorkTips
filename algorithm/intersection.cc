#include <iostream>
#include <vector>
#include <map>

// N个有序数组，找到在所有数组的交际
// **

void do_cpp17()
{
  std::vector< std::vector<int> > a{
    {3, 4, 5, 6, 8},
    {0, 2, 3, 4, 5, 8},
    {0, 1, 4, 8},
    {1, 2, 3, 4, 5, 6, 7, 8}
  };

  int N = 4;

  std::vector<int> array_size;
  for(auto v:a)
    array_size.push_back(v.size());

  int min = array_size[0];
  int index = 0;
  for(int i = 0; i < N; i++){
    if(min > array_size[i] ){
      min = array_size[i];
      index = i;
    }
  }

  // left_max and right_min
  int left_max = a[0][0];
  int right_min = a[0][array_size[0]-1];
  for(int i = 1; i < N; i++){
    if(left_max < a[i][0])
      left_max = a[i][0];
    if(right_min > a[i][array_size[i]-1])
      right_min = a[i][array_size[i]-1];
  }

  std::vector<int> common;
  for(int i = 0; i < array_size[index]; i++ ){
    if(a[index][i] < left_max || a[index][i] > right_min)
      break;
    int target = a[index][i];


    for(int j = 0; j < N; j++){
      for(int k = 0; k < a[j][array_size[j]-1]; k++){
        if(a[j][k] == target){
          a[j].pop();
          continue;
        }
      }
    }

  }
  





  return;
}

int main()
{
  do_cpp17();
  
  return 0;
}
