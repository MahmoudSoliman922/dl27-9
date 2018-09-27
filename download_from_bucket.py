#!/usr/bin/python
import boto3
s3=boto3.client('s3')
list=s3.list_objects(Bucket='face-rec-final')['Contents']
for key in list:
    s3.download_file('face-rec-final', key['Key'], 'data/'+key['Key'])