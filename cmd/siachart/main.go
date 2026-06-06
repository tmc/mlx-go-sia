// Command siachart renders a SIA run's throughput series for the demo. It reads
// each generation's results.json (written by the ThroughputEvaluator) from a
// run tree and emits a terminal sparkline+table for the live view and, with
// -csv, a CSV for gnuplot or a spreadsheet. It works for any run that writes the
// shared results.json schema — P3 (cmd/inferopt) and P7 (cmd/metalopt) alike.
//
// Usage:
//
//	siachart -run runs/run_1                  # terminal chart of the delta series
//	siachart -runs-root runs -run-id 1        # same run, addressed by id
//	siachart -run runs/run_1 -metric speedup  # plot speedup instead of delta
//	siachart -run runs/run_1 -csv chart.csv   # also write CSV for gnuplot
//	siachart -run runs/run_1 -csv -           # CSV to stdout
//
// The reported metric is the gen-N − gen-0 delta the evaluator already computed
// (cancelling thermal/cache drift); the Y-axis unit is read from the results
// (tokens/sec for P3, ops/sec for P7), never hardcoded.
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strconv"

	"github.com/tmc/mlx-go-sia/cmd/siachart/internal/chartdata"
)

func main() {
	fs := flag.NewFlagSet("siachart", flag.ExitOnError)
	var (
		runDir   = fs.String("run", "", "path to a run directory (e.g. runs/run_1); contains gen_N/results.json")
		runsRoot = fs.String("runs-root", "runs", "runs root, used with -run-id when -run is unset")
		runID    = fs.Int("run-id", -1, "run id under -runs-root (alternative to -run)")
		metric   = fs.String("metric", "delta", "series to plot: delta | speedup | tokens")
		csvOut   = fs.String("csv", "", "also write CSV here; \"-\" for stdout")
	)
	fs.Parse(os.Args[1:])

	dir := *runDir
	if dir == "" {
		if *runID < 0 {
			fmt.Fprintln(os.Stderr, "siachart: provide -run <dir> or -run-id <n>")
			os.Exit(2)
		}
		dir = filepath.Join(*runsRoot, "run_"+strconv.Itoa(*runID))
	}

	m, err := parseMetric(*metric)
	if err != nil {
		fmt.Fprintln(os.Stderr, "siachart:", err)
		os.Exit(2)
	}

	series, err := chartdata.ReadSeries(dir)
	if err != nil {
		fmt.Fprintln(os.Stderr, "siachart:", err)
		os.Exit(1)
	}

	// When CSV goes to stdout the terminal chart goes to stderr, so the piped
	// CSV stays clean; otherwise the chart is the primary stdout output.
	chartOut := os.Stdout
	if *csvOut == "-" {
		chartOut = os.Stderr
	}
	if err := chartdata.RenderTerminal(series, m, chartOut); err != nil {
		fmt.Fprintln(os.Stderr, "siachart: render:", err)
		os.Exit(1)
	}

	if *csvOut != "" {
		if err := writeCSV(series, *csvOut); err != nil {
			fmt.Fprintln(os.Stderr, "siachart: csv:", err)
			os.Exit(1)
		}
		if *csvOut != "-" {
			fmt.Printf("\nwrote CSV: %s\n", *csvOut)
		}
	}
}

func parseMetric(s string) (chartdata.Metric, error) {
	switch s {
	case "delta", "":
		return chartdata.MetricDelta, nil
	case "speedup":
		return chartdata.MetricSpeedup, nil
	case "tokens", "throughput":
		return chartdata.MetricTokensPerSec, nil
	default:
		return 0, fmt.Errorf("unknown -metric %q (delta|speedup|tokens)", s)
	}
}

func writeCSV(s chartdata.Series, path string) error {
	if path == "-" {
		return chartdata.WriteCSV(s, os.Stdout)
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return chartdata.WriteCSV(s, f)
}
