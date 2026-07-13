//go:build darwin

package main

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/tmc/cc"
	"github.com/tmc/cc/cass"
	"github.com/tmc/cc/cass/collector"
)

// PiAgg is the aggregate rollup across every discovered pi session. Every field
// is a real fold of parsed pi data; nothing is invented. Tokens come from the
// collector's per-session SessionStats.
//
// OutputTokensReliable is false when any session's output tokens came from the
// JSONL streaming-start snapshot (or a BPE estimate) rather than a final count.
// pi sessions today carry only the snapshot, so the panel labels output tokens
// as approximate rather than presenting them as authoritative.
type PiAgg struct {
	Found        bool // a pi sessions dir exists and was scanned
	Dir          string
	Sessions     int
	InputTokens  int
	OutputTokens int
	CacheTokens  int
	ToolCalls    int
	Turns        int
	ToolBreak    map[string]int // tool name -> count, folded across sessions

	OutputTokensReliable bool
}

// PiEntry is one row of the live activity feed: a single parsed pi entry reduced
// to what the panel shows. Summary is the tool name for tool blocks, or the
// first ~SummaryLen chars of text otherwise.
type PiEntry struct {
	When    time.Time
	HasTime bool
	Role    string // user, assistant, or the entry type when there is no message
	Kind    string // text, tool_use, tool_result, or the raw entry type
	Tool    string // tool name when Kind is tool_use/tool_result, else ""
	Summary string // short human-readable summary
	Session string // short session title for context
}

// PiSnapshot is one consistent read of pi state: the aggregate strip plus the
// newest activity rows, newest last.
type PiSnapshot struct {
	Agg   PiAgg
	Feed  []PiEntry
	Error string // non-empty when scanning failed outright
}

const (
	summaryLen  = 60 // chars of text to keep in an activity summary
	maxFeedRows = 24 // newest entries shown in the live feed
)

// pollPi scans the pi sessions under dir (or the default ~/.pi/agent/sessions
// when dir is empty) and returns an aggregate rollup plus the newest activity
// rows. It never fabricates: an empty dir yields Agg.Found=true with zero
// sessions and an empty feed, which the UI renders as an explicit "no pi runs
// found" state.
func pollPi(ctx context.Context, dir string) PiSnapshot {
	pi := &collector.Pi{Root: dir}

	det, _ := pi.Detect(ctx)
	agg := PiAgg{ToolBreak: map[string]int{}, OutputTokensReliable: true}
	if det != nil {
		agg.Found = det.Found
		if len(det.Paths) > 0 {
			agg.Dir = det.Paths[0]
		}
	}
	if !agg.Found {
		return PiSnapshot{Agg: agg}
	}

	scanCtx, cancel := context.WithTimeout(ctx, 4*time.Second)
	defer cancel()

	out := make(chan cass.Session)
	scanErr := make(chan error, 1)
	go func() { scanErr <- pi.Scan(scanCtx, cass.ScanConfig{}, out) }()

	var sessions []cass.Session
	for s := range out {
		sessions = append(sessions, s)
	}
	if err := <-scanErr; err != nil && len(sessions) == 0 {
		return PiSnapshot{Agg: agg, Error: err.Error()}
	}

	// Fold the per-session stats the collector already computed.
	agg.Sessions = len(sessions)
	for _, s := range sessions {
		st := s.Stats
		agg.InputTokens += st.InputTokens
		agg.OutputTokens += st.OutputTokensSnapshot
		agg.CacheTokens += st.CacheReads + st.CacheCreationInputTokens
		agg.ToolCalls += st.ToolCalls
		agg.Turns += st.Turns
		for name, n := range st.ToolBreakdown {
			agg.ToolBreak[name] += n
		}
		if st.OutputTokensEstimated {
			agg.OutputTokensReliable = false
		}
	}
	// pi JSONL only carries the streaming-start output snapshot, never a final
	// count, so output tokens are approximate whenever there is any output.
	if agg.OutputTokens > 0 {
		agg.OutputTokensReliable = false
	}

	feed := buildFeed(ctx, sessions)
	return PiSnapshot{Agg: agg, Feed: feed}
}

// buildFeed reads raw entries from the most-recently-ended sessions and reduces
// them to feed rows, newest last, capped at maxFeedRows. It re-reads via
// cc.ReadFile so the feed reflects entry-level activity (the collector's
// cass.Message view drops tool-only and non-message entries we want to show).
func buildFeed(ctx context.Context, sessions []cass.Session) []PiEntry {
	// Newest session first so we tail the freshest activity.
	ordered := append([]cass.Session(nil), sessions...)
	sort.Slice(ordered, func(i, j int) bool {
		return ordered[i].EndedAt.After(ordered[j].EndedAt)
	})

	var rows []PiEntry
	for _, s := range ordered {
		entries, err := cc.ReadFile(ctx, s.SourcePath)
		if err != nil {
			continue
		}
		title := strings.TrimSpace(s.Title)
		if title == "" {
			title = shortID(s.ID)
		}
		for _, e := range entries {
			rows = append(rows, entryRow(e, title))
		}
		if len(rows) >= maxFeedRows {
			break
		}
	}

	// Sort all collected rows chronologically; entries without a timestamp sort
	// to the front so dated activity reads newest-last at the bottom.
	sort.SliceStable(rows, func(i, j int) bool {
		if rows[i].HasTime != rows[j].HasTime {
			return !rows[i].HasTime // undated first
		}
		return rows[i].When.Before(rows[j].When)
	})
	if len(rows) > maxFeedRows {
		rows = rows[len(rows)-maxFeedRows:]
	}
	return rows
}

// entryRow reduces one parsed pi entry to a feed row. Tool blocks become
// "tool_use:<name>"; text becomes the first ~summaryLen chars. Non-message
// entries (session_meta, model_change, ...) keep their type as the kind.
func entryRow(e cc.Entry, sessionTitle string) PiEntry {
	row := PiEntry{
		When:    e.Timestamp,
		HasTime: !e.Timestamp.IsZero(),
		Kind:    e.Type,
		Session: sessionTitle,
	}
	if e.Message == nil {
		// CLI/system entry: model_change, session_meta, progress, etc.
		switch {
		case e.Content != "":
			row.Summary = truncateText(e.Content)
		case e.Summary != "":
			row.Summary = truncateText(e.Summary)
		default:
			row.Summary = e.Type
		}
		return row
	}

	row.Role = e.Message.Role
	blocks := e.Message.ContentBlocks()
	for _, b := range blocks {
		switch b.Type {
		case "tool_use":
			row.Kind, row.Tool = "tool_use", b.Name
			row.Summary = "tool_use: " + b.Name
			return row
		case "tool_result":
			row.Kind = "tool_result"
			res := strings.TrimSpace(b.Content)
			if res == "" {
				res = "(result)"
			}
			row.Summary = "tool_result: " + truncateText(res)
			return row
		}
	}
	// No tool block: fall back to text.
	text := strings.TrimSpace(e.Message.TextContent())
	if text == "" {
		row.Kind = e.Type
		row.Summary = "(empty " + e.Message.Role + " message)"
		return row
	}
	row.Kind = "text"
	row.Summary = truncateText(text)
	return row
}

func truncateText(s string) string {
	s = strings.TrimSpace(strings.Join(strings.Fields(s), " "))
	if len(s) > summaryLen {
		return s[:summaryLen] + "…"
	}
	return s
}

func shortID(id string) string {
	if len(id) > 8 {
		return id[:8]
	}
	return id
}

// piFingerprint is a cheap change token for the pi snapshot: the poller compares
// it between ticks and only triggers a rebuild when the data actually moves
// (a session appears/disappears, tokens change, or new activity lands).
func piFingerprint(s PiSnapshot) string {
	var b strings.Builder
	fmt.Fprintf(&b, "found=%v;dir=%s;n=%d;in=%d;out=%d;cache=%d;tools=%d;turns=%d;err=%s|",
		s.Agg.Found, s.Agg.Dir, s.Agg.Sessions, s.Agg.InputTokens, s.Agg.OutputTokens,
		s.Agg.CacheTokens, s.Agg.ToolCalls, s.Agg.Turns, s.Error)
	names := make([]string, 0, len(s.Agg.ToolBreak))
	for name := range s.Agg.ToolBreak {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		fmt.Fprintf(&b, "%s=%d;", name, s.Agg.ToolBreak[name])
	}
	b.WriteByte('|')
	for _, e := range s.Feed {
		fmt.Fprintf(&b, "%d/%s/%s;", e.When.UnixNano(), e.Kind, e.Summary)
	}
	return b.String()
}
