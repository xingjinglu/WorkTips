// Largest 1-Bordered square
#include<vector>
#include<iostream>

struct xy{int x, y;};

int get_max_rect(std::vector<std::vector<int> >&grid)
{
  int res = 0; // number of elements.
  int rows = grid.size();
  int cols = grid[0].size();
  printf("rows = %d, cols = %d \n",rows, cols);

  std::vector<std::vector<xy>> memo(rows+1, std::vector<xy>(cols+1, {0, 0}));

  for(int i = 1; i <= rows ; i++){
    for(int j = 1; j <= cols; j++){

      if(grid[i-1][j-1] == 1){
        memo[i][j].y = memo[i][j-1].y + 1;
        memo[i][j].x = memo[i-1][j].x + 1;

        int mn =  std::min(memo[i][j].x,  memo[i][j].y);

        for(int k = 0; k < mn - 1; k++){
          if(memo[i][j-(mn-1)+k].x >= (mn-k) && memo[i-(mn-1)+k][j].y >= (mn-k)){
            res = std::max(res, (mn-k)*(mn-k)); 
            break;
          }
        }
        res = std::max(res, 1);
      }
    }
  }

  return res;
}


int main()
{
  std::vector<std::vector<int>> grid = {{1, 1, 1}, {1, 0, 1}, {1, 1, 1}};

  int res = get_max_rect(grid);
  printf("res = %d \n", res);

  return 0;
}
