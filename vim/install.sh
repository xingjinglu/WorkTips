set -xe

export https_proxy=http://10.130.14.129:8080
export http_proxy=http://10.130.14.129:8080

# Upgrade vim.
sudo yum install ncurses-devel
wget https://github.com/vim/vim/archive/master.zip
unzip master.zip
cd vim-master/src/
./configure && make -j64 && make install

sudo mv /usr/bin/vim /usr/bin/vim.bak
sudo ln /usr/local/bin/vim /usr/bin -s


# Install pathogen  
# ~/.vim/bundle是pathogen默认runtimepath，把所有的plugin放到该目录即可
mkdir -p ~/.vim/autoload ~/.vim/bundle  && curl -LSso ~/.vim/autoload/pathogen.vim https://tpo.pe/pathogen.vim

cat > ~/.vimrc <<EOF
execute pathogen#infect()
syntax on
filetype plugin indent on
EOF



# Install vim-plugin  

cd ~/.vim/bundle
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
#vim t
#:Helptags  # 自动生成所有plugin的文档

cat >> ~/.bashrc <<EOF
mkdir -p  ~/software/go_workspace
export GOPATH=~/software/go_workspace
export GOROOT=/usr/lib/golang # 默认安装目录
export PATH=$PATH:$GOPATH/bin
EOF



# 配置vim-go,会自动从网上下载相应包
curl -fLo ~/.vim/autoload/plug.vim --create-dirs https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim
git clone https://github.com/fatih/vim-go.git ~/.vim/plugged/vim-go

go get -u github.com/jstemmer/gotags
go get -u github.com/mdempsky/gocode

# Install go-package, need internet.
#vim t
#::GoInstallBinaries

cat >> ~/.vimrc <<EOF
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
EOF





