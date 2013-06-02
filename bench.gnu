set title "Bench"
set xlabel "Iteration"
set ylabel "value"
set yrange [0:1000]
set xrange [0:0.03] 
plot "benchMass.dat" using 1:2 with lines
set term png 
set output "fig.png"
