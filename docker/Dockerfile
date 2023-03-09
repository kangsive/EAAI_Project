FROM python:3.9.11

# 初始设置
WORKDIR /root/bakery

# 构建Python环境  
COPY ./requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -U setuptools \
    && pip install --no-cache-dir -r requirements.txt

CMD uwsgi --ini uwsgi.ini
