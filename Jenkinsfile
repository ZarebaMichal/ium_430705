pipeline{
    agent any
    properties([parameters([text(defaultValue: '50', description: 'Number of lines to cutoff', name: 'CUTOFF')])])
    stages{
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
            withEnv(["CUTOFF=${params.CUTOFF}"])
                ./script.sh
            }
        }
        stage('Archive artifacts'){
            steps{
                archiveArtifacts 'test.csv'
                archiveArtifacts 'dev.csv'
                archiveArtifacts 'train.csv'
            }
        }
    }
}