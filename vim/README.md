
- Install go
该目录最好是用户私有目录，不然后面配置vim-go 比较麻烦
export gopath=~/software/go_workspace
export goroot=/usr/lib/golang # 默认安装目录
export PATH=$PATH:$gopath/bin


# 配置vim工具 with pathogen

- update vim  

```
sudo yum install ncurses-devel
wget https://github.com/vim/vim/archive/master.zip
unzip master.zip
cd vim-master/src/
./configure && make -j64 && make install

# 注意该vim会安装到/usr/local/bin ，但是系统自带vim可能在/usr/bin下，所以要做对应的软链接
```

- Install pathogen  
```
# ~/.vim/bundle是pathogen默认runtimepath，把所有的plugin放到该目录即可
mkdir -p ~/.vim/autoload ~/.vim/bundle  && curl -LSso ~/.vim/autoload/pathogen.vim https://tpo.pe/pathogen.vim

# 修改.vimrc
execute pathogen#infect()
syntax on
filetype plugin indent on

```

- Install vim-plugin  

```
# cd ~/.vim/bundle
git clone https://github.com/majutsushi/tagbar.git
git clone https://github.com/scrooloose/nerdtree.git
git clone https://github.com/powerline/powerline.git
git clone https://github.com/fatih/vim-go.git
git clone https://github.com/tpope/vim-sensible.git
git clone https://github.com/Shougo/neocomplete.vim.git
git clone https://github.com/sjl/gundo.vim
git clone https://github.com/Blackrush/vim-gocode.git
git clone https://github.com/plasticboy/vim-markdown.git

# Generate help docs
vim t
:Helptags  # 自动生成所有plugin的文档

```

- 配置vim-go,会自动从网上下载相应包
```
go get -u github.com/jstemmer/gotags
go get -u github.com/mdempsky/gocode

vim t
::GoInstallBinaries

# cat .vimrc
"golang                                                                                                                                             
let g:tagbar_type_go = {                                                                                                                            
  \ 'ctagstype' : 'go',
  \ 'kinds'     : [
    \ 'p:package', 
    \ 'i:imports:1',
    \ 'c:constants',
    \ 'v:variables',
    \ 't:types', 
    \ 'n:interfaces',                                                                                                                               
    \ 'w:fields',                                                                                                                                   
    \ 'e:embedded',
    \ 'm:methods',                                                                                                                                  
    \ 'r:constructor',                                                                                                                              
    \ 'f:functions'
  \ ],
  \ 'sro' : '.',
  \ 'kind2scope' : {                                                                                                                                
    \ 't' : 'ctype',                                                                                                                                
    \ 'n' : 'ntype'
  \ },
  \ 'scope2kind' : {                                                                                                                                
    \ 'ctype' : 't',
    \ 'ntype' : 'n'                                                                                                                                 
  \ },
  \ 'ctagsbin'  : 'gotags',
  \ 'ctagsargs' : '-sort -silent'
\ }

```




