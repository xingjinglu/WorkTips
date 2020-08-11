// Largest 1-Bordered square
#include<vector>
#include<iostream>

int get_max_rect(std::vector<std::vector<int> >&grid)
{
  int res = 0; // number of elements.
  int rows = grid.size();
  int cols = grid[0].size();
  printf("rows = %d, cols = %d \n",rows, cols);

  std::vector<std::vector<int>> memo(rows, std::vector<int>(cols, 0));

  for(int i = 0; i < rows; i++){
    if(grid[i][0] == 1){
      memo[i][0] = 1;
      res = 1;
    }  
  }

  for(int i = 0; i < cols; i++){
    if(grid[0][i] == 1){
      memo[0][i] = 1;
      res = 1;
    }  
  }


  for(int i = 1; i < rows ; i++){
    for(int j = 1; j < cols; j++){
      if(grid[i][j] == 1){
        memo[i][j] = std::min(memo[i-1][j], std::min(memo[i-1][j-1], memo[i][j-1])) + 1;
        res = std::max(res, memo[i][j]);
      }
    }
  }

  return res * res;
}


int main()
{
  std::vector<std::vector<int>> grid = {{1, 1, 1}, {1, 0, 1}, {1, 1, 1}};

  int res = get_max_rect(grid);
  printf("res = %d \n", res);

  return 0;
}
