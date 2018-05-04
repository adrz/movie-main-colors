FROM julianbei/alpine-opencv-microimage:p3-3.1

WORKDIR /

COPY entrypoint.sh /
COPY ./main-colors-docker.py /app/
COPY ./main-colors.py /app/
COPY ./src /app/src/
COPY requirements.txt /

RUN pip install --upgrade pip && \
    pip install virtualenv && \
    virtualenv -p python3 env && \
    . env/bin/activate && \
    pip install -r requirements.txt

ENTRYPOINT ["/entrypoint.sh"]