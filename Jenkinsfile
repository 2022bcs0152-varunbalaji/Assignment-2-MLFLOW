pipeline {
    agent any

    stages {

        stage('Install') {
            steps {
                sh 'pip install -r requirements.txt'
            }
        }

        stage('Train Model') {
            steps {
                sh 'python src/train.py'
            }
        }

        stage('Build Docker') {
            steps {
                sh 'docker build -t churn-mlops .'
            }
        }

        stage('Run Container') {
            steps {
                sh 'docker run -d -p 8000:8000 churn-mlops'
            }
        }
    }
}