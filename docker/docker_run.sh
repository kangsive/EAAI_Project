setupdir="/home/dingkang/codes/EAAI_Project"

docker run --name=bakery -p 0.0.0.0:8851:8851 -it -d \
        -v $setupdir:/root/bakery \
         big_bakery:1.0 uwsgi --ini /root/bakery/uwsgi.ini