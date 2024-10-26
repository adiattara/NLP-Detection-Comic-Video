# MLFlow Docker Setup [![Actions Status](https://github.com/Toumash/mlflow-docker/workflows/VerifyDockerCompose/badge.svg)](https://github.com/Toumash/mlflow-docker/actions)

> If you want to boot up mlflow project with one-liner - this repo is for you. 
> The only requirement is docker installed on your system and we are going to use Bash on linux/windows.

# 🚀 1-2-3! Setup guide 
1. Configure `.env` file for your choice. You can put there anything you like, it will be used to configure you services
2. Run `docker compose up`
3. Open up http://localhost:5000 for MlFlow, and http://localhost:9001/ to browse your files in S3 artifact store

# Features
 - One file setup (.env)
 - Minio S3 artifact store with GUI
 - MySql mlflow storage
 - Ready to use bash scripts for python development!
 - Automatically-created s3 buckets
