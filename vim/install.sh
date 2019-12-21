set -xe

export https_proxy=http://10.130.14.129:8080
export http_proxy=http://10.130.14.129:8080

# Upgrade vim.
sudo yum install ncurses-devel
wget https://github.com/vim/vim/archive/master.zip
unzip master.zip
cd vim-master/src/
./configure && make -j64 && make install

rm vim-master master.zip -rf

sudo mv /usr/bin/vim /usr/bin/vim.bak
sudo ln /usr/local/bin/vim /usr/bin -s


# Install pathogen  
# ~/.vim/bundle是pathogen默认runtimepath，把所有的plugin放到该目录即可
mkdir -p ~/.vim/autoload ~/.vim/bundle  && curl -LSso ~/.vim/autoload/pathogen.vim https://tpo.pe/pathogen.vim


# Install vim-plugin  
git clone https://github.com/majutsushi/tagbar.git   ~/.vim/bundle/tagbar
git clone https://github.com/scrooloose/nerdtree.git  ~/.vim/bundle/nerdtree
git clone https://github.com/powerline/powerline.git ~/.vim/bundle/powerline
git clone https://github.com/fatih/vim-go.git  ~/.vim/bundle/vim-go
git clone https://github.com/tpope/vim-sensible.git ~/.vim/vim-sensible
git clone https://github.com/Shougo/neocomplete.vim.git ~/.vim/neocomplete.vim
git clone https://github.com/sjl/gundo.vim  ~/.vim/gundo.vim
git clone https://github.com/Blackrush/vim-gocode.git  ~/.vim/vim-gocode
git clone https://github.com/plasticboy/vim-markdown.git ~/.vim/vim-markdown


mkdir -p  ~/software/go_workspace
cat >> ~/.bashrc <<EOF
export GOPATH=/search/odin/xxx/software/go_workspace
export GOROOT=/usr/lib/golang # 默认安装目录
export PATH=$PATH:$GOPATH/bin
EOF
source ~/.bashrc



# 配置vim-go,会自动从网上下载相应包
curl -fLo ~/.vim/autoload/plug.vim --create-dirs https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim
git clone https://github.com/fatih/vim-go.git ~/.vim/plugged/vim-go

go get -u github.com/jstemmer/gotags
go get -u github.com/mdempsky/gocode

mv ~/.vimrc ~/.vimrc.bak
cat >  ~/.vimrc <<EOF
execute pathogen#infect()
syntax on
filetype plugin indent on

" cscope related
set nocscopeverbose  

set mouse=ar
set hlsearch
colorscheme desert
set smartindent
set expandtab
set shiftwidth=2
set tabstop=2
set autoindent

syntax on
set hlsearch
set showmatch



let Tlist_Auto_Open = 0
let Tlist_Auto_Update = 0

set cscopetag
set textwidth=80


set nocompatible
set t_Co=256

set wildmenu
set wildmode=longest:list,full

set completeopt=longest,menu,preview
set backspace=indent,eol,start

set history=200

set laststatus=2
set showcmd
set incsearch
set splitright
set splitbelow

au FileType python set expandtab shiftwidth=4 tabstop=4

" OpenCL format.
autocmd BufNewFile,BufRead *.cl set filetype=opencl
autocmd FileType opencl source /search/speech/luxingjing/.vim/plugin/opencl.vim

"NerdTree
nnoremap <silent> <F2> :NERDTree <CR>

"TagbarToggle
nnoremap <silent> <F7> :TagbarToggle<CR>
let TagbarOpenAutoClose = 0
let tagbar_autoclose = 0
let tagbar_autoopen = 1

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



# Install other stuffs.


# Generate help docs
#vim t
#:Helptags  # 自动生成所有plugin的文档

pushd ~/.vim/bundle
echo "vim t"
echo ":GoInstallBinaries"

