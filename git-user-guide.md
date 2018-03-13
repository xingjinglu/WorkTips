
#


## 1. 合并多次commit

### git rebase 
rebase 是将上游更改合并进本地仓库的通常方法，不要用在公共分支上。 

```
# 开始新的功能分支
git checkout -b new-feature master
# 编辑文件
git commit -a -m "Start developing a feature"

// 为了消除master的bug，该过程产生分叉
# 基于master分支创建一个快速修复分支
git checkout -b hotfix master
# 编辑文件
git commit -a -m "Fix security hole"
# 合并回master
git checkout master
git merge hotfix
git branch -d hotfix

// 消除分叉的历史
git checkout new-feature
git rebase master

//  快速前向合并
git checkout master
git merge new-feature

```

- git rebase -i   
```
git rebase -i  xxxx // 第一个不想合并的commit id

如果出错  
git rebase --abort // 恢复到最初状态
```
- 问题：对于已经push的commit是否可行？ （好像不可以）  


### git commit 
-  修复前一次commit（未push）   
修复最新提交,可以修改最近一次提交并且可以修改 message，但是不会产生新的提交记录。 但是如果，上次提交已经push，则该命令失效。  
```
git commit --amend   
```


### git revert   
```
git revert --abort
``` 


###
