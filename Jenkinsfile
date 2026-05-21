pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                git branch: 'master', url: 'https://github.com/rahulnavi26/Sign-Language-for-speech-conversion.git'
            }
        }

        stage('Check Python') {
            steps {
                sh 'python3 --version'
                sh 'pip3 --version'
            }
        }

        stage('Create Virtual Environment') {
            steps {
                sh 'rm -rf venv'
                sh 'python3 -m venv venv'
            }
        }

        stage('Install Dependencies') {
            steps {
                sh './venv/bin/python -m pip install --upgrade pip'
                sh './venv/bin/pip install -r requirements.txt'
            }
        }

        stage('Run Python File') {
            steps {
                sh './venv/bin/python Testing.py'
            }
        }
    }
}
