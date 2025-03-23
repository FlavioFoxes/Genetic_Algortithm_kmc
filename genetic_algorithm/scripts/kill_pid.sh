#!/bin/bash

# Prende il PID del processo che usa la porta 12345 
# (usare anche le altre porte è ridondante). 
# Quindi uccidi quel processo
PID=$(lsof -t -i :12345)
kill $PID
echo "BASH  ->  Processo ucciso"

# Uccidi SimRobot 
PID_SIM=$(pidof SimRobot)
kill $PID_SIM