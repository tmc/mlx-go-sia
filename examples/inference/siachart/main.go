// Command siachart renders a SIA run's throughput series for the demo. It reads
// each generation's results.json (written by the ThroughputEvaluator) from a
// run tree and emits a terminal sparkline+table for the live view and, with
// -csv, a CSV for gnuplot or a spreadsheet. It works for any run that writes the
// shared results.json schema — P3 (inferopt) and P7 (metalopt) alike.
//
// Usage:
//
//	siachart -run runs/run_1                  # terminal chart of the delta series
//	siachart -runs-root runs -run-id 1        # same run, addressed by id
//	siachart -run runs/run_1 -metric speedup  # plot speedup instead of delta
//	siachart -run runs/run_1 -metric correctness  # live-cloud gate progression
//	siachart -run runs/run_1 -csv chart.csv   # also write CSV for gnuplot
//	siachart -run runs/run_1 -csv -           # CSV to stdout
//
// The reported metric is the gen-N − gen-0 delta the evaluator already computed
// (cancelling thermal/cache drift); the Y-axis unit is read from the results
// (tokens/sec for P3, ops/sec for P7), never hardcoded.
//
// siachart also reads a weights run tree, written by the WeightsEvaluator, whose
// metric is a held-out test_loss where lower is better. The schema is detected
// from the results.json, so a weights tree renders its own view — quality bars
// that rise with the loss plus a signed Δ-vs-best column — without a flag.
//
// With -metric correctness, a throughput tree is rendered as the live-cloud
// gate progression: the verdict/correctness arc (REVISE the model's real bugs,
// PASS the correct candidates) with speed shown only when a generation's
// candidate and baseline distributions separate — otherwise labelled parity, so
// a near-1.0x speedup is never drawn as a win.
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strconv"

	"github.com/tmc/mlx-go-sia/examples/inference/siachart/internal/chartdata"
)

func main() {
	fs := flag.NewFlagSet("siachart", flag.ExitOnError)
	var (
		runDir   = fs.String("run", "", "path to a run directory (e.g. runs/run_1); contains gen_N/results.json")
		runsRoot = fs.String("runs-root", "runs", "runs root, used with -run-id when -run is unset")
		runID    = fs.Int("run-id", -1, "run id under -runs-root (alternative to -run)")
		metric   = fs.String("metric", "delta", "series to plot: delta | speedup | tokens | correctness")
		model    = fs.String("model", "", "model label for the correctness view header (e.g. the cloud model name)")
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

	// When CSV goes to stdout the terminal chart goes to stderr, so the piped
	// CSV stays clean; otherwise the chart is the primary stdout output.
	chartOut := os.Stdout
	if *csvOut == "-" {
		chartOut = os.Stderr
	}

	// -metric correctness selects the live-cloud correctness-progression view: a
	// throughput tree, but rendered as the gate's record (REVISE the bugs, PASS
	// the correct candidates) with speed shown only when the distributions
	// separate. It is opt-in because the tree's schema is the throughput one; the
	// view is a choice of story, not a different file format.
	if *metric == "correctness" {
		if err := runNebius(dir, *model, chartOut, *csvOut); err != nil {
			fmt.Fprintln(os.Stderr, "siachart:", err)
			os.Exit(1)
		}
		return
	}

	m, err := parseMetric(*metric)
	if err != nil {
		fmt.Fprintln(os.Stderr, "siachart:", err)
		os.Exit(2)
	}

	// The run tree declares its own schema: a weights tree reports a held-out
	// test_loss (lower is better), a throughput tree reports tokens/sec (higher
	// is better). Detect it from the results.json rather than a flag, so the data
	// can never disagree with how it is plotted. -metric applies to throughput
	// only; a weights tree has a single view.
	if isWeights, ok := chartdata.IsWeightsTree(dir); ok && isWeights {
		if err := runWeights(dir, chartOut, *csvOut); err != nil {
			fmt.Fprintln(os.Stderr, "siachart:", err)
			os.Exit(1)
		}
		return
	}

	series, err := chartdata.ReadSeries(dir)
	if err != nil {
		fmt.Fprintln(os.Stderr, "siachart:", err)
		os.Exit(1)
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

// runNebius reads and renders a live-cloud run tree as a correctness progression
// (the gate's record: REVISE the bugs, PASS the correct candidates, credit a
// speedup only when the distributions separate).
func runNebius(dir, model string, chartOut *os.File, csvOut string) error {
	series, err := chartdata.ReadNebiusSeries(dir)
	if err != nil {
		return err
	}
	series.Model = model
	if err := series.Render(chartOut); err != nil {
		return fmt.Errorf("render: %w", err)
	}
	if csvOut != "" {
		if csvOut == "-" {
			return series.WriteCSV(os.Stdout)
		}
		f, err := os.Create(csvOut)
		if err != nil {
			return fmt.Errorf("csv: %w", err)
		}
		defer f.Close()
		if err := series.WriteCSV(f); err != nil {
			return fmt.Errorf("csv: %w", err)
		}
		fmt.Printf("\nwrote CSV: %s\n", csvOut)
	}
	return nil
}

// runWeights reads and renders a weights run tree (held-out test_loss series).
func runWeights(dir string, chartOut *os.File, csvOut string) error {
	series, err := chartdata.ReadWeightsSeries(dir)
	if err != nil {
		return err
	}
	if err := series.Render(chartOut); err != nil {
		return fmt.Errorf("render: %w", err)
	}
	if csvOut != "" {
		if csvOut == "-" {
			return series.WriteCSV(os.Stdout)
		}
		f, err := os.Create(csvOut)
		if err != nil {
			return fmt.Errorf("csv: %w", err)
		}
		defer f.Close()
		if err := series.WriteCSV(f); err != nil {
			return fmt.Errorf("csv: %w", err)
		}
		fmt.Printf("\nwrote CSV: %s\n", csvOut)
	}
	return nil
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
