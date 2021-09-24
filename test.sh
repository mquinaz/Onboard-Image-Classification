(./ica.py -H > output.txt 2>&1)&
sleep 5
./icmsg.py -m autoML_slow -s 10 setup
./icmsg.py start
sleep 10
./icmsg.py stop
./icmsg.py setup
sleep 1
killall ica.py
grep -e posix -e svmem output.txt
