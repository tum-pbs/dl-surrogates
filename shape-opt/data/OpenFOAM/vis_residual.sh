#!/bin/bash

foamLog 2_foam.log >/dev/null

gnuplot -persist > /dev/null 2>&1 << EOF
      set logscale y
      set title "Residual vs. Iteration"
      set xlabel "Iteration"
      set ylabel "Residual"
      plot "logs/Ux_0" with lines,\
           "logs/Uy_0" with lines,\
           "logs/p_0" with lines,\
           "logs/nuTilda_0" with lines

EOF
