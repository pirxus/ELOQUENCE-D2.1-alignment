#!/bin/bash
#$ -N ssd_cleanup
#$ -q short.q@supergpu5*
#$ -l ram_free=1G,mem_free=1G
#$ -l ssd=1,ssd_free=20G
#$ -o /mnt/matylda6/xsedla1h/projects/job_logs/ssd_cleanup/cleanup.o
#$ -e /mnt/matylda6/xsedla1h/projects/job_logs/ssd_cleanup/cleanup.e
#

ls /mnt/ssd/xsedla1h/

echo "Cleaning the ssd directory.."
rm -rf /mnt/ssd/xsedla1h/*
