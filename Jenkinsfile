pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                git branch: 'master', url: 'https://github.com/rahulnavi26/Sign-Language-for-speech-conversion.git'
            }
        }

        stage('Install Dependencies') {
            steps {
                bat 'python --version'
                bat 'pip install -r requirements.txt'
            }
        }

        stage('Run Python File') {
            steps {
                bat 'python Testing.py'
            }
        }
    }
}
