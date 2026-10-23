pipeline {
    agent any

    options {
        timestamps()
        timeout(time: 20, unit: 'MINUTES')
        buildDiscarder(logRotator(numToKeepStr: '10'))
    }

    triggers {
        // Fallback if the GitHub webhook cannot reach Jenkins (e.g. Jenkins on localhost)
        pollSCM('H/2 * * * *')
    }

    environment {
        APP_URL = 'https://iris-classifier-afhj9thfmxxyeczdchmkdh.streamlit.app/'
        VENV    = '.venv'
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
                sh 'git log -1 --oneline'
            }
        }

        stage('Setup Environment') {
            steps {
                sh '''
                    python3 -m venv ${VENV}
                    ${VENV}/bin/pip install --upgrade pip
                    ${VENV}/bin/pip install -r requirements-ci.txt
                '''
            }
        }

        stage('Test') {
            steps {
                sh '''
                    mkdir -p reports
                    ${VENV}/bin/python -m pytest --junitxml=reports/junit.xml
                '''
            }
            post {
                always {
                    junit allowEmptyResults: true, testResults: 'reports/junit.xml'
                }
            }
        }

        stage('Train Model (MLflow)') {
            steps {
                sh '${VENV}/bin/python train.py'
            }
        }

        stage('Archive Artifacts') {
            steps {
                archiveArtifacts artifacts: 'model.pkl, reports/*.xml', fingerprint: true
            }
        }

        stage('Deploy') {
            when {
                expression { env.GIT_BRANCH == 'origin/main' || env.BRANCH_NAME == 'main' }
            }
            steps {
                echo 'Streamlit Community Cloud redeploys automatically when main is pushed to GitHub.'
                // Smoke test of the live app; a failure marks the build UNSTABLE, not FAILED
                catchError(buildResult: 'SUCCESS', stageResult: 'UNSTABLE') {
                    sh 'curl -fsSL --max-time 60 -o /dev/null -w "Live app HTTP status: %{http_code}\\n" "${APP_URL}"'
                }
            }
        }
    }

    post {
        success { echo 'Pipeline finished: tests passed, model trained and archived.' }
        failure { echo 'Pipeline failed: check the console output of the failing stage.' }
        cleanup { cleanWs() }
    }
}
