package sia

import (
	"io"
	"os"
	"strings"
)

// newTokenReplacer builds the substituter for an [ExecRunner] argument list. The
// recognized tokens are:
//
//	%MODEL%      the model the engine should drive
//	%MAXTURNS%   the turn budget
//	%WORKDIR%    the generation working directory
//	%BASEURL%    the provider's OpenAI-compatible base URL (empty for native)
//	%APIKEY_ENV% the name of the env var holding the provider's API key
//	%APIKEY%     the API key value, read from %APIKEY_ENV% at substitution time
//
// %BASEURL%/%APIKEY_ENV%/%APIKEY% let an OpenAI-compatible engine CLI be pointed
// at a provider (e.g. Nebius Token Factory) without hardcoding the endpoint.
func newTokenReplacer(req AgentRequest) *strings.Replacer {
	apiKey := ""
	if req.Provider.APIKeyEnv != "" {
		apiKey = os.Getenv(req.Provider.APIKeyEnv)
	}
	return strings.NewReplacer(
		"%MODEL%", req.Model,
		"%MAXTURNS%", itoa(req.MaxTurns),
		"%WORKDIR%", req.WorkingDir,
		"%BASEURL%", req.Provider.BaseURL,
		"%APIKEY_ENV%", req.Provider.APIKeyEnv,
		"%APIKEY%", apiKey,
	)
}

func stringReader(s string) io.Reader { return strings.NewReader(s) }

func orStdout(f *os.File) io.Writer {
	if f == nil {
		return os.Stdout
	}
	return f
}

func orStderr(f *os.File) io.Writer {
	if f == nil {
		return os.Stderr
	}
	return f
}

// teeProgress returns a writer that fans subprocess output out to the log file
// and the in-memory buffer, plus an optional live progress sink. A nil progress
// sink (the default) means the executor stays quiet — output still reaches the
// log and buffer — so library callers and tests see no console spam; a CLI sets
// progress to os.Stdout to watch a long-running generation as it runs.
func teeProgress(logFile, buf io.Writer, progress io.Writer) io.Writer {
	if progress == nil {
		return io.MultiWriter(logFile, buf)
	}
	return io.MultiWriter(logFile, buf, progress)
}
