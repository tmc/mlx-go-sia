# Plot a siachart CSV as the demo's climbing-throughput chart with error bars.
#
#   siachart -run runs/run_1 -csv series.csv
#   gnuplot -e "csv='series.csv'; out='chart.png'" cmd/siachart/plot.gnuplot
#
# Plots the gen-N - gen-0 delta (median) with candidate min/max as error bars.
# REVISE generations have empty numeric cells and so appear as gaps. The Y-axis
# unit comes from the data (tokens/sec for P3, ops/sec for P7); set ylabel to
# match via -e "unit='ops/sec'".

set datafile separator ","
set key off
set grid ytics
set xlabel "generation"
if (!exists("unit")) unit = "throughput/sec"
set ylabel sprintf("gen-N − gen-0 delta (%s)", unit)
set title "SIA self-improvement: throughput climbing across generations"

if (exists("out")) {
    set terminal pngcairo size 900,540 font ",12"
    set output out
}

# Columns (1-indexed): 1 gen ... 8 delta_tokens_per_sec ... 10 cand_min 11 cand_median 12 cand_max
# Error bars use candidate spread around the median delta is not directly the
# candidate spread, so we plot candidate median with its min/max whiskers and
# the delta line together.
plot csv using 1:8 with linespoints lw 2 pt 7 ps 1.4 title "delta", \
     csv using 1:11:10:12 with yerrorbars lw 1 pt 0 title "cand spread"
