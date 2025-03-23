#!/bin/bash

# Funzione per lanciare SimRobot in background e aspettare risposta da esso
function run_simRobot {
    # Compilazione ed esecuzione
    # touch /home/flavio/Scrivania/RoboCup/spqrnao2024/Make/CMake/Nao.cmake            # notice new symbols in new files
    # NO_CLION=true /home/flavio/Scrivania/RoboCup/spqrnao2024/Make/Linux/generate     # notice new files
    # /home/flavio/Scrivania/RoboCup/spqrnao2024/Make/Linux/compile                    # compile!
    echo "BASH  ->   Lancio SimRobot..."
    $PWD/../../Build/Linux/SimRobot/Develop/SimRobot $PWD/../../Config/Scenes/DescriptionFiles/Fast/BH/[Fast]BH_Kick.ros2 &    # Lancia SimRobot in background
    simrobot_pid=$!

    # Dopo aver lanciato SimRobot:
    #   1) Per aspettare la risposta da SimRobot, creo un file temporaneo dove salvare il messaggio ricevuto
    #   2) Il messaggio viene ricevuto ascoltando in UDP sulla porta 12345, dove il comando per l'ascolto (netcat)
    #      viene eseguito in background (altrimenti lo script si blocca sull'ascolto).
    #   3) Poi il messaggio viene letto dal file temporaneo
    #   4) A questo punto uccido il processo di netcat
    #   5) Uccido il processo di SimRobot
    #   6) Mando pacchetto all'environment per sincronizzare i processi
    echo "BASH  ->   Aspettando risposta da SimRobot..."
    
    # 1)
    TEMP_FILE=$(mktemp)
    trap 'rm -f $TEMP_FILE' EXIT

    # 2)
    echo "BASH  ->   In ascolto sulla porta 12345 (UDP)..."
    nc -u -l 12345 > $TEMP_FILE &
    NC_PID=$!

    # 3)
    while true; do
        if [ -s $TEMP_FILE ]; then
            MESSAGE=$(cat $TEMP_FILE)
            echo "BASH  ->   Ricevuto da SimRobot: $MESSAGE"
            break
        fi
        # sleep 1
    done

    # 4)
    kill $NC_PID

    # 5)
    echo "BASH  ->   Chiudo SimRobot..."
    kill $simrobot_pid

    # 6)
    echo "Killed Sim" | nc -u 127.0.0.1 5432  # mando messaggio di chiusura a ENV
}

# Funzione per aspettare risposta da client
function wait_for_client {
    echo "Aspettando risposta da client..."
    TEMP_FILE=$(mktemp)
    trap 'rm -f $TEMP_FILE' EXIT

    # 2)
    # echo "In ascolto sulla porta 12346 (UDP)..."
    nc -u -l 12346 > $TEMP_FILE &
    NC_PID=$!

    # 3)
    while true; do
        if [ -s $TEMP_FILE ]; then
            MESSAGE=$(cat $TEMP_FILE)
            echo "Risposta ricevuta da client: $MESSAGE"
            break
        fi
        sleep 1
    done

    # 4)
    kill $NC_PID
}

# Ciclo principale
run_simRobot
