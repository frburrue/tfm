FROM docker.elastic.co/beats/filebeat:7.6.2

USER root
COPY filebeat.yml /usr/share/filebeat/filebeat.yml
RUN chown root:filebeat /usr/share/filebeat/filebeat.yml && chmod 0644 /usr/share/filebeat/filebeat.yml