FROM raspbian/stretch:latest

WORKDIR /app

RUN apt-get update && apt-get upgrade -y \
	&& apt-get install -y --no-install-recommends lsb-release curl wget build-essential \
        && rm -rf /var/lib/apt/lists/*

RUN wget -qO install.sh https://raw.githubusercontent.com/node-red/linux-installers/master/deb/update-nodejs-and-nodered # https://raw.githubusercontent.com/node-red/raspbian-deb-package/master/resources/update-nodejs-and-nodered
RUN chmod +x install.sh && yes | ./install.sh
RUN rm install.sh

COPY entrypoint.sh .
COPY settings.js /root/.node-red

ENTRYPOINT ["/bin/bash", "entrypoint.sh"]

