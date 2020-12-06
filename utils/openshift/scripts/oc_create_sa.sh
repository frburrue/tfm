#!/bin/bash

services=(backend worker filebeat)

for service in ${services[@]}; do

	oc create sa ${service}
	oc adm policy add-scc-to-user anyuid -z ${service}

done
