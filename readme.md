Setting up the environment:

1) Before running the nvidia docker, you'll need to install NVIDIA Container Toolkit. 
Refer to this page: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker

2) Then run the following commands to run my docker:

xhost +
sudo docker run --gpus all -it --rm --net=host --privileged -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY --mount type=bind,src=<path/to/DeepstreamSolution>,dst=/DeepstreamSolution cpeeris/deepstreamsolutiondocker

3) Setup mysql (https://phoenixnap.com/kb/install-mysql-ubuntu-20-04)

Step 1: Update/Upgrade Package Repository
sudo apt update
sudo apt upgrade

Step 2: Install MySQL
sudo apt install mysql-server
mysql --version

Step 3: Securing MySQL
sudo mysql_secure_installation
- enter and renter password (we use Password!23)

Step 4: Check if MySQL Service Is Running
sudo systemctl status mysql

Step 5: Log in to MySQL Server
sudo mysql -u root


4) Install RabbitMQ

Step 1: Install Erlang

sudo apt update
sudo apt install curl software-properties-common apt-transport-https lsb-release
curl -fsSL https://packages.erlang-solutions.com/ubuntu/erlang_solutions.asc | sudo gpg --dearmor -o /etc/apt/trusted.gpg.d/erlang.gpg
echo "deb https://packages.erlang-solutions.com/ubuntu $(lsb_release -cs) contrib" | sudo tee /etc/apt/sources.list.d/erlang.list
sudo apt update
sudo apt install erlang

Step 2: Add RabbitMQ Repository to Ubuntu

curl -s https://packagecloud.io/install/repositories/rabbitmq/rabbitmq-server/script.deb.sh | sudo bash

Step 3: Install RabbitMQ Server 
sudo apt update
sudo apt install rabbitmq-server
systemctl status rabbitmq-server.service
systemctl is-enabled rabbitmq-server.service

If the service is disabled, enable it:
sudo systemctl enable rabbitmq-server

