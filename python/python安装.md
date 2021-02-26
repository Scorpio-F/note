# python安装



### 安装依赖，下载源码包

```shell
# CentOs 7
yum install libffi-devel wget sqlite-devel xz gcc atuomake zlib-devel openssl-devel epel-release git -y
wget https://www.python.org/ftp/python/3.9.0/Python-3.9.0.tgz

```



### 解压

```shell
tar -zxvf Python-3.9.0.tgz
cd Python-3.9.0/

```



### 编译安装

```shell
./configure --prefix=/usr/local/python3
./configure --enable-optimizations
make && make install

```



### 添加软链接

```shell
ln -s /usr/local/python3/bin/python3 /usr/bin/python3
ln -s /usr/local/python3/bin/pip3 /usr/bin/pip3

```



### 添加环境变量

```shell
vim /etc/profile
export PATH=$PATH:/usr/local/python3/bin
source /etc/profile

```



### 安装完成