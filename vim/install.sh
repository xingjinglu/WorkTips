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


# 配置vim-go,会自动从网上下载相应包
go get -u github.com/jstemmer/gotags
go get -u github.com/mdempsky/gocode

#
#vim t
#::GoInstallBinaries



