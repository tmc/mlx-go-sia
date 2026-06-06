package sia

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestParseTrainSpec checks that a train.py is read as a declarative spec: only
// whitelisted keys are taken, arbitrary code and unknown keys are ignored, and a
// whitelisted key with a bad value is a parse error (not silently dropped).
func TestParseTrainSpec(t *testing.T) {
	tests := []struct {
		name    string
		src     string
		want    TrainSpec
		wantErr bool
	}{
		{
			name: "all whitelisted keys",
			src: `learning_rate = 0.0003
lora_rank = 16
num_layers = 8
iters = 50
batch_size = 2
fine_tune_type = "lora"
data_mix = "lawbench"`,
			want: TrainSpec{
				LearningRate: ptr(0.0003),
				LoRARank:     ptr(16),
				NumLayers:    ptr(8),
				Iters:        ptr(50),
				BatchSize:    ptr(2),
				FineTuneType: "lora",
				DataMix:      "lawbench",
			},
		},
		{
			name: "colon assignment and comments stripped",
			src: `# tuning config
learning_rate: 1e-4  # small LR
lora_rank: 8`,
			want: TrainSpec{LearningRate: ptr(1e-4), LoRARank: ptr(8)},
		},
		{
			name: "arbitrary code and unknown keys ignored",
			src: `import os
os.system("rm -rf /")          # must never run
secret = open("/etc/passwd").read()
epochs = 99                    # not whitelisted
    iters = 9999               # indented: nested code, ignored
learning_rate = 0.001`,
			want: TrainSpec{LearningRate: ptr(0.001)},
		},
		{
			name: "trailing comma (dict-style) tolerated",
			src:  `iters = 25,`,
			want: TrainSpec{Iters: ptr(25)},
		},
		{
			name:    "non-numeric learning_rate is an error",
			src:     `learning_rate = "fast"`,
			wantErr: true,
		},
		{
			name:    "non-integer iters is an error",
			src:     `iters = 3.5`,
			wantErr: true,
		},
		{
			name: "empty spec is not an error",
			src:  "print('hello')\n# nothing to see here",
			want: TrainSpec{},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parseTrainSpec([]byte(tt.src))
			if (err != nil) != tt.wantErr {
				t.Fatalf("parseTrainSpec err = %v, wantErr %v", err, tt.wantErr)
			}
			if tt.wantErr {
				return
			}
			if !specEqual(got, tt.want) {
				t.Errorf("parseTrainSpec = %s, want %s", specStr(got), specStr(tt.want))
			}
		})
	}
}

// TestBuildArgs checks the spec→flag translation: defaults applied, agent spec
// overrides them, -train is always present, the adapter writes into the gen dir,
// and -data points at DataDir — never the held-out test set.
func TestBuildArgs(t *testing.T) {
	const model = "mlx-community/Qwen3-0.6B-4bit"
	const dataDir = "/data/lawbench" // train/valid only

	tests := []struct {
		name     string
		exec     MLXTrainExecutor
		spec     TrainSpec
		wantArgs map[string]string // flag -> value
		wantErr  bool
	}{
		{
			name: "defaults when spec empty",
			exec: MLXTrainExecutor{BaseModel: model, DataDir: dataDir},
			spec: TrainSpec{},
			wantArgs: map[string]string{
				"-model":          model,
				"-data":           dataDir,
				"-fine-tune-type": "lora",
				"-lora-rank":      "8",
				"-num-layers":     "16",
				"-iters":          "100",
				"-batch-size":     "4",
				"-learning-rate":  "1e-05",
				"-adapter-path":   "adapters",
			},
		},
		{
			name: "agent spec overrides defaults",
			exec: MLXTrainExecutor{BaseModel: model, DataDir: dataDir},
			spec: TrainSpec{
				LearningRate: ptr(0.0003),
				LoRARank:     ptr(32),
				NumLayers:    ptr(4),
				Iters:        ptr(50),
				BatchSize:    ptr(8),
				FineTuneType: "dora",
			},
			wantArgs: map[string]string{
				"-learning-rate":  "0.0003",
				"-lora-rank":      "32",
				"-num-layers":     "4",
				"-iters":          "50",
				"-batch-size":     "8",
				"-fine-tune-type": "dora",
			},
		},
		{
			name: "executor defaults fill unset spec fields",
			exec: MLXTrainExecutor{BaseModel: model, DataDir: dataDir, Defaults: TrainSpec{Iters: ptr(200), LoRARank: ptr(64)}},
			spec: TrainSpec{LoRARank: ptr(16)}, // spec beats default
			wantArgs: map[string]string{
				"-iters":     "200", // from Defaults
				"-lora-rank": "16",  // from spec, overriding Defaults
			},
		},
		{
			name:     "data_mix selects a subdirectory",
			exec:     MLXTrainExecutor{BaseModel: model, DataDir: dataDir},
			spec:     TrainSpec{DataMix: "subsetA"},
			wantArgs: map[string]string{"-data": filepath.Join(dataDir, "subsetA")},
		},
		{
			name:    "data_mix path escape rejected",
			exec:    MLXTrainExecutor{BaseModel: model, DataDir: dataDir},
			spec:    TrainSpec{DataMix: "../test"},
			wantErr: true,
		},
		{
			name:    "invalid fine_tune_type rejected",
			exec:    MLXTrainExecutor{BaseModel: model, DataDir: dataDir},
			spec:    TrainSpec{FineTuneType: "qlora"},
			wantErr: true,
		},
		{
			name:    "zero iters rejected",
			exec:    MLXTrainExecutor{BaseModel: model, DataDir: dataDir},
			spec:    TrainSpec{Iters: ptr(0)},
			wantErr: true,
		},
		{
			name:    "negative learning_rate rejected",
			exec:    MLXTrainExecutor{BaseModel: model, DataDir: dataDir},
			spec:    TrainSpec{LearningRate: ptr(-0.1)},
			wantErr: true,
		},
		{
			name:    "zero lora_rank rejected",
			exec:    MLXTrainExecutor{BaseModel: model, DataDir: dataDir},
			spec:    TrainSpec{LoRARank: ptr(0)},
			wantErr: true,
		},
		{
			name:     "num_layers -1 (all) allowed",
			exec:     MLXTrainExecutor{BaseModel: model, DataDir: dataDir},
			spec:     TrainSpec{NumLayers: ptr(-1)},
			wantArgs: map[string]string{"-num-layers": "-1"},
		},
		{
			name:    "num_layers -2 rejected",
			exec:    MLXTrainExecutor{BaseModel: model, DataDir: dataDir},
			spec:    TrainSpec{NumLayers: ptr(-2)},
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			args, err := tt.exec.buildArgs(tt.spec)
			if (err != nil) != tt.wantErr {
				t.Fatalf("buildArgs err = %v, wantErr %v", err, tt.wantErr)
			}
			if tt.wantErr {
				return
			}
			flags := flagMap(args)
			if !hasFlag(args, "-train") {
				t.Errorf("missing -train flag; args = %v", args)
			}
			for flag, want := range tt.wantArgs {
				if got := flags[flag]; got != want {
					t.Errorf("%s = %q, want %q\nargs = %v", flag, got, want, args)
				}
			}
			// Honesty: the held-out test set must never be referenced.
			if strings.Contains(strings.Join(args, " "), "test.jsonl") {
				t.Errorf("args reference test data: %v", args)
			}
		})
	}
}

// TestRunTargetSuccess runs RunTarget against a fake mlx-lm-train that records
// its argv and exits 0, confirming success and that combined output is logged.
func TestRunTargetSuccess(t *testing.T) {
	requireSh(t)
	dir := t.TempDir()
	argvFile := filepath.Join(dir, "argv.txt")
	bin := fakeTrainBin(t, dir, 0, argvFile, "training complete")

	work := filepath.Join(dir, "gen_1")
	mustMkdir(t, work)
	specPath := filepath.Join(work, "train.py")
	writeTestFile(t, specPath, "iters = 5\nlearning_rate = 0.001\n")
	logPath := filepath.Join(work, "train_stdout.log")

	e := &MLXTrainExecutor{TrainBin: bin, BaseModel: "m", DataDir: filepath.Join(dir, "data")}
	res, err := e.RunTarget(context.Background(), TargetRequest{
		AgentPath: specPath, DatasetDir: filepath.Join(dir, "data"), WorkingDir: work, StdoutLog: logPath,
	})
	if err != nil {
		t.Fatalf("RunTarget: %v", err)
	}
	if !res.Success {
		t.Errorf("Success = false, want true; ErrorMsg=%q", res.ErrorMsg)
	}
	if !strings.Contains(res.Stdout, "training complete") {
		t.Errorf("Stdout missing trainer output: %q", res.Stdout)
	}
	// The translated flags reached the trainer.
	argv, _ := os.ReadFile(argvFile)
	for _, want := range []string{"-train", "-iters", "5", "-learning-rate", "0.001"} {
		if !strings.Contains(string(argv), want) {
			t.Errorf("trainer argv missing %q\ngot:\n%s", want, argv)
		}
	}
	// Combined output was written to the stdout log too.
	logged, _ := os.ReadFile(logPath)
	if !strings.Contains(string(logged), "training complete") {
		t.Errorf("stdout log missing trainer output: %q", logged)
	}
}

// TestRunTargetTrainerFails confirms a non-zero trainer exit is reported as
// feedback (Success:false, ErrorMsg set) and NOT as a Go error.
func TestRunTargetTrainerFails(t *testing.T) {
	requireSh(t)
	dir := t.TempDir()
	bin := fakeTrainBin(t, dir, 1, filepath.Join(dir, "argv.txt"), "out of memory")

	work := filepath.Join(dir, "gen_1")
	mustMkdir(t, work)
	specPath := filepath.Join(work, "train.py")
	writeTestFile(t, specPath, "iters = 5\n")

	e := &MLXTrainExecutor{TrainBin: bin, BaseModel: "m", DataDir: filepath.Join(dir, "data")}
	res, err := e.RunTarget(context.Background(), TargetRequest{
		AgentPath: specPath, WorkingDir: work, StdoutLog: filepath.Join(work, "train_stdout.log"),
	})
	if err != nil {
		t.Fatalf("RunTarget returned a Go error for a trainer non-zero exit: %v", err)
	}
	if res.Success {
		t.Error("Success = true, want false on trainer failure")
	}
	if res.ErrorMsg == "" {
		t.Error("ErrorMsg empty, want a failure message")
	}
}

// TestRunTargetMissingSpec confirms a missing train.py is feedback, not a Go error.
func TestRunTargetMissingSpec(t *testing.T) {
	dir := t.TempDir()
	work := filepath.Join(dir, "gen_1")
	mustMkdir(t, work)
	e := &MLXTrainExecutor{TrainBin: "/bin/true", BaseModel: "m", DataDir: dir}
	res, err := e.RunTarget(context.Background(), TargetRequest{
		AgentPath:  filepath.Join(work, "train.py"), // never created
		WorkingDir: work, StdoutLog: filepath.Join(work, "train_stdout.log"),
	})
	if err != nil {
		t.Fatalf("RunTarget: %v", err)
	}
	if res.Success || res.ErrorMsg == "" {
		t.Errorf("missing spec: got Success=%v ErrorMsg=%q, want Success=false with message", res.Success, res.ErrorMsg)
	}
}

// TestRunTargetMissingBinary confirms that an unrunnable trainer is reported as a
// Go error (the executor itself cannot run), per the TargetExecutor contract.
func TestRunTargetMissingBinary(t *testing.T) {
	dir := t.TempDir()
	work := filepath.Join(dir, "gen_1")
	mustMkdir(t, work)
	specPath := filepath.Join(work, "train.py")
	writeTestFile(t, specPath, "iters = 5\n")

	e := &MLXTrainExecutor{TrainBin: filepath.Join(dir, "does-not-exist"), BaseModel: "m", DataDir: dir}
	res, err := e.RunTarget(context.Background(), TargetRequest{
		AgentPath: specPath, WorkingDir: work, StdoutLog: filepath.Join(work, "train_stdout.log"),
	})
	if err == nil {
		t.Error("RunTarget err = nil, want a Go error when the trainer binary is missing")
	}
	if res.Success {
		t.Error("Success = true, want false when the trainer binary is missing")
	}
}

// TestRunTargetRejectsTestData confirms the honesty guard: if the trainer's data
// directory contains the held-out test.jsonl, the run is refused as feedback
// (Success:false) rather than silently training on the eval rows.
func TestRunTargetRejectsTestData(t *testing.T) {
	requireSh(t)
	dir := t.TempDir()
	data := filepath.Join(dir, "data")
	mustMkdir(t, data)
	writeTestFile(t, filepath.Join(data, "train.jsonl"), "{}\n")
	writeTestFile(t, filepath.Join(data, "test.jsonl"), "{}\n") // must trip the guard

	work := filepath.Join(dir, "gen_1")
	mustMkdir(t, work)
	specPath := filepath.Join(work, "train.py")
	writeTestFile(t, specPath, "iters = 5\n")

	e := &MLXTrainExecutor{TrainBin: "/bin/true", BaseModel: "m", DataDir: data}
	res, err := e.RunTarget(context.Background(), TargetRequest{
		AgentPath: specPath, WorkingDir: work, StdoutLog: filepath.Join(work, "train_stdout.log"),
	})
	if err != nil {
		t.Fatalf("RunTarget: %v", err)
	}
	if res.Success {
		t.Error("Success = true, want false when DataDir contains test.jsonl")
	}
	if !strings.Contains(res.ErrorMsg, "test.jsonl") {
		t.Errorf("ErrorMsg = %q, want it to mention test.jsonl", res.ErrorMsg)
	}
}

// TestRunTargetContextCancelled confirms that a cancelled context is reported as
// feedback (Success:false, no Go error), not as an executor failure.
func TestRunTargetContextCancelled(t *testing.T) {
	requireSh(t)
	dir := t.TempDir()
	// A trainer that blocks long enough for the cancellation to land.
	bin := filepath.Join(dir, "sleeper.sh")
	writeTestFile(t, bin, "#!/bin/sh\nsleep 30\n")
	if err := os.Chmod(bin, 0o755); err != nil {
		t.Fatal(err)
	}
	work := filepath.Join(dir, "gen_1")
	mustMkdir(t, work)
	specPath := filepath.Join(work, "train.py")
	writeTestFile(t, specPath, "iters = 5\n")

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // already cancelled before RunTarget launches the trainer
	e := &MLXTrainExecutor{TrainBin: bin, BaseModel: "m", DataDir: filepath.Join(dir, "data")}
	res, err := e.RunTarget(ctx, TargetRequest{
		AgentPath: specPath, WorkingDir: work, StdoutLog: filepath.Join(work, "train_stdout.log"),
	})
	if err != nil {
		t.Fatalf("RunTarget returned a Go error for a cancelled context: %v", err)
	}
	if res.Success {
		t.Error("Success = true, want false on cancellation")
	}
	if !strings.Contains(res.ErrorMsg, "cancel") {
		t.Errorf("ErrorMsg = %q, want it to mention cancellation", res.ErrorMsg)
	}
}

// TestRunTargetRequiresConfig confirms missing BaseModel/DataDir is a Go error.
func TestRunTargetRequiresConfig(t *testing.T) {
	for _, e := range []*MLXTrainExecutor{
		{DataDir: "/d"},  // no BaseModel
		{BaseModel: "m"}, // no DataDir
	} {
		_, err := e.RunTarget(context.Background(), TargetRequest{})
		if err == nil {
			t.Errorf("RunTarget(%+v) err = nil, want a configuration error", e)
		}
	}
}

func TestSandboxLocal(t *testing.T) {
	if SandboxLocal == SandboxModal || SandboxLocal == SandboxFusion {
		t.Errorf("SandboxLocal %q collides with a stock sandbox value", SandboxLocal)
	}
	if !IsLocalSandbox(SandboxLocal) || IsLocalSandbox(SandboxModal) {
		t.Error("IsLocalSandbox should be true only for SandboxLocal")
	}
}

// --- test helpers ---

func ptr[T any](v T) *T { return &v }

func requireSh(t *testing.T) {
	t.Helper()
	if _, err := os.Stat("/bin/sh"); err != nil {
		t.Skip("no /bin/sh")
	}
}

func mustMkdir(t *testing.T, dir string) {
	t.Helper()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
}

// fakeTrainBin writes an executable shell script that mimics mlx-lm-train: it
// records its argv to argvFile, prints msg, and exits with code. It keeps tests
// hermetic — no real training runs.
func fakeTrainBin(t *testing.T, dir string, code int, argvFile, msg string) string {
	t.Helper()
	bin := filepath.Join(dir, "fake-mlx-lm-train.sh")
	script := "#!/bin/sh\nprintf '%s\\n' \"$@\" > '" + argvFile + "'\n" +
		"echo '" + msg + "'\nexit " + itoa(code) + "\n"
	writeTestFile(t, bin, script)
	if err := os.Chmod(bin, 0o755); err != nil {
		t.Fatal(err)
	}
	return bin
}

// flagMap pairs each -flag with its following value. A token is a flag name
// only if it starts with '-' followed by a letter, so a negative-number value
// like "-1" (dash + digit) is treated as a value, not a flag.
func flagMap(args []string) map[string]string {
	isFlagName := func(s string) bool {
		return len(s) >= 2 && s[0] == '-' && (s[1] < '0' || s[1] > '9')
	}
	m := make(map[string]string)
	for i := 0; i+1 < len(args); i++ {
		if isFlagName(args[i]) && !isFlagName(args[i+1]) {
			m[args[i]] = args[i+1]
		}
	}
	return m
}

func hasFlag(args []string, flag string) bool {
	for _, a := range args {
		if a == flag {
			return true
		}
	}
	return false
}

func specEqual(a, b TrainSpec) bool {
	return eqFloatPtr(a.LearningRate, b.LearningRate) &&
		eqIntPtr(a.LoRARank, b.LoRARank) &&
		eqIntPtr(a.NumLayers, b.NumLayers) &&
		eqIntPtr(a.Iters, b.Iters) &&
		eqIntPtr(a.BatchSize, b.BatchSize) &&
		a.FineTuneType == b.FineTuneType &&
		a.DataMix == b.DataMix
}

func eqIntPtr(a, b *int) bool {
	if a == nil || b == nil {
		return a == b
	}
	return *a == *b
}

func eqFloatPtr(a, b *float64) bool {
	if a == nil || b == nil {
		return a == b
	}
	return *a == *b
}

func specStr(s TrainSpec) string {
	var b strings.Builder
	b.WriteString("{")
	if s.LearningRate != nil {
		b.WriteString("lr=" + formatFloat(*s.LearningRate) + " ")
	}
	if s.LoRARank != nil {
		b.WriteString("rank=" + itoa(*s.LoRARank) + " ")
	}
	if s.NumLayers != nil {
		b.WriteString("layers=" + itoa(*s.NumLayers) + " ")
	}
	if s.Iters != nil {
		b.WriteString("iters=" + itoa(*s.Iters) + " ")
	}
	if s.BatchSize != nil {
		b.WriteString("batch=" + itoa(*s.BatchSize) + " ")
	}
	if s.FineTuneType != "" {
		b.WriteString("ft=" + s.FineTuneType + " ")
	}
	if s.DataMix != "" {
		b.WriteString("mix=" + s.DataMix + " ")
	}
	b.WriteString("}")
	return b.String()
}
