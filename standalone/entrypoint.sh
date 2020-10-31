#!/bin/bash

echo "nameserver 8.8.8.8\n" >> /etc/resolv.conf
python app.py
