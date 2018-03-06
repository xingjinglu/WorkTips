
#


## 1. 合并多次commit

### git rebase （合并多次commit）
```
git rebase -i  xxxx // 第一个不想合并的commit id

如果出错  
git rebase --abort // 恢复到最初状态
```
- 问题：对于已经push的commit是否可行？ （好像不可以）  


### git commit 
- 将当前缓存与前一次commit合并    
修复最新提交,可以修改上次提交的 message，但是不会产生新的提交记录。 但是如果，上次提交已经push，则该命令失效。  
```
git commit --amend 
```


### git revert   
```
git revert --abort
``` 


###