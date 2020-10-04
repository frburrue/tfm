#!/bin/bash

tar xvf openshift-origin-server-v3.11.0-0cbc58b-linux-64bit.tar.gz
cd openshift-origin-server-v3.11.0-0cbc58b-linux-64bit
sudo cp oc kubectl /usr/local/bin/
cd ..
sudo sh -c 'echo "{ \"insecure-registries\": [\"172.30.0.0/16\"] } " > /etc/docker/daemon.json'
sudo service docker restart
