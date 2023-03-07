docker stop bakery
docker rm bakery
docker rmi big_bakery:1.0
docker build -t big_bakery:1.0 .
sudo bash docker_run.sh