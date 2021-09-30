(./ica.py -H > output.txt 2>&1)&
sleep 10
./icmsg.py -m $1 -s $2 setup
sleep 1
./icmsg.py start
sleep $3
./icmsg.py stop
sleep 1
killall ica.py
grep CPU output.txt
