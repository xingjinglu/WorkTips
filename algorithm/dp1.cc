#include <iostream>
 
#define N 20        //台阶数为20
using namespace std;
 
int dp[N];          //全局数组，存放决策表
 
int fun(int n)      //返回台阶数为n的走法
{
	if (n == 1 || n == 2)
	{
		return n;
	}
	dp[n-1] = fun(n-1);        //若不为1或2则进行递归计算
	dp[n-2] = fun(n-2);
	dp[n] = dp[n-1]+dp[n-2];   //状态转移方程
	return dp[n];
}
 
int main(int argc,char** argv)
{
	fun(N);
	cout<<"15: " << dp[15]<<endl;        //输出15阶的走法
                             //
	//system("pause");
	return 0;
}

