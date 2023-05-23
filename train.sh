#!/bin/bash
for j in 'D37' 'D46' 'D55' ;
  do
  for i in 100 300 600 800 ;
  do
  python main.py --mode train --D $j --model dgl --epochs $i ;
  python main.py --mode train --D $j --model gru --epochs $i ;
#  python main.py --mode train --D $j --model pattern --epochs $i ;
  python main.py --mode train --D $j --model all --epochs $i ;
  python main.py --mode train --D $j --model all_double --epochs $i --double grup ;
  python main.py --mode train --D $j --model all_double --epochs $i --double gcnp ;
  python main.py --mode train --D $j --model all_double --epochs $i --double gg ;
  done
done