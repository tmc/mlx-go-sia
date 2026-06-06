package sia

import (
	"fmt"
	"os"
	"path/filepath"
	"slices"
)

// TaskLayout builds the paths for a resolved task directory and its shared
// directory. The zero value is not usable; construct with [NewTaskLayout] or
// [ResolveTaskDir].
type TaskLayout struct {
	TaskDir   string
	SharedDir string
}

// NewTaskLayout returns the layout for a task directory and its shared dir.
func NewTaskLayout(taskDir, sharedDir string) TaskLayout {
	return TaskLayout{TaskDir: taskDir, SharedDir: sharedDir}
}

// DatasetDir returns the task's public dataset directory (the read-only data
// the target agent sees via --dataset_dir).
func (t TaskLayout) DatasetDir() string {
	return filepath.Join(t.TaskDir, NameDataPublic)
}

// AbsDatasetDir returns the absolute path of [TaskLayout.DatasetDir].
func (t TaskLayout) AbsDatasetDir() string {
	abs, err := filepath.Abs(t.DatasetDir())
	if err != nil {
		return t.DatasetDir()
	}
	return abs
}

// TaskMD returns the path to the task description (data/public/task.md).
func (t TaskLayout) TaskMD() string {
	return filepath.Join(t.TaskDir, NameTaskMD)
}

// SampleDescriptions returns the path to the sample task descriptions.
func (t TaskLayout) SampleDescriptions() string {
	return filepath.Join(t.TaskDir, NameSampleTaskDescriptions)
}

// ReferenceDir returns the task's reference directory.
func (t TaskLayout) ReferenceDir() string {
	return filepath.Join(t.TaskDir, NameReferenceDir)
}

// ReferenceAgent returns the path to the task's bundled reference agent.
func (t TaskLayout) ReferenceAgent() string {
	return filepath.Join(t.TaskDir, NameReferenceAgent)
}

// SampleExecution returns the path to the shared sample execution trajectory.
func (t TaskLayout) SampleExecution() string {
	return filepath.Join(t.SharedDir, NameSharedSampleExecution)
}

// EvaluateScript locates evaluate.py: it prefers data/public/evaluate.py, then
// task_dir/evaluate.py, returning the empty string if neither exists.
func (t TaskLayout) EvaluateScript() string {
	candidate := filepath.Join(t.TaskDir, NameDataPublic, NameEvaluatePy)
	if isFile(candidate) {
		return candidate
	}
	candidate = filepath.Join(t.TaskDir, NameEvaluatePy)
	if isFile(candidate) {
		return candidate
	}
	return ""
}

// ResolveTaskDir resolves a task name or an explicit task directory into a
// (taskDir, sharedDir) pair. Exactly one of task/taskDir must be non-empty.
//
//   - task name: tasksRoot/<task>/, shared = tasksRoot/_shared/. The name must be
//     one of [BundledTasks] and the directory must exist under tasksRoot.
//   - taskDir: the directory itself, shared = taskDir/../_shared/ if present else
//     tasksRoot/_shared/.
//
// tasksRoot is where bundled tasks live (the reference embeds them in its wheel;
// the Go port takes the root as an argument so tasks are plain directories).
func ResolveTaskDir(tasksRoot, task, taskDir string) (TaskLayout, error) {
	bundledShared := filepath.Join(tasksRoot, NameSharedDir)

	switch {
	case task != "" && taskDir != "":
		return TaskLayout{}, fmt.Errorf("provide only one of task or task_dir")
	case task != "":
		if !slices.Contains(BundledTasks, task) {
			return TaskLayout{}, fmt.Errorf("unknown task %q: available: %s", task, joinSorted(BundledTasks))
		}
		resolved := filepath.Join(tasksRoot, task)
		if !isDir(resolved) {
			return TaskLayout{}, fmt.Errorf("task %q not found under %s", task, tasksRoot)
		}
		return TaskLayout{TaskDir: resolved, SharedDir: bundledShared}, nil
	case taskDir != "":
		resolved, err := filepath.Abs(taskDir)
		if err != nil {
			return TaskLayout{}, err
		}
		if !isDir(resolved) {
			return TaskLayout{}, fmt.Errorf("task directory does not exist: %s", taskDir)
		}
		shared := filepath.Join(filepath.Dir(resolved), NameSharedDir)
		if !isDir(shared) {
			shared = bundledShared
		}
		return TaskLayout{TaskDir: resolved, SharedDir: shared}, nil
	default:
		return TaskLayout{}, fmt.Errorf("either task or task_dir must be provided")
	}
}

func isDir(path string) bool {
	info, err := os.Stat(path)
	return err == nil && info.IsDir()
}
