pipeline{
    agent any
    parameters {
    string(defaultValue: '5', description: 'Amount of lines to cut off the file.', name: 'CUTOFF', trim: false)
}
    stages{
        stage('Test') {
            agent {
                dockerfile true
            }
            steps {
                sh 'python3 ./script2.py'
            }
      }
        stage('Stage 1'){
            steps{
                echo 'Hello World!'
            }
        }
        stage('checkout: Check out from version control'){
            steps{
                checkout([$class: 'GitSCM', branches: [[name: '*/master']], doGenerateSubmoduleConfigurations: false, extensions: [], submoduleCfg: [], userRemoteConfigs: [[credentialsId: '3e2b5ea1-7a08-4418-892b-c562315036a4', url: 'https://git.wmi.amu.edu.pl/s430705/ium_430705']]])
            }
        }
        stage('sh: Shell Script'){
            steps{
            withEnv(["CUTOFF=${params.CUTOFF}"]) {
                sh "chmod 777 ./script.sh"
                sh "python3 ./script2.py"
                archiveArtifacts 'test.csv'
                archiveArtifacts 'dev.csv'
                archiveArtifacts 'train.csv'
            }
        }}
        stage('Archive artifacts'){
            steps{


        }
    }
}}