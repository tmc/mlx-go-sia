//go:build darwin

package main

import (
	"encoding/json"
	"fmt"
	"os"
	"sync"
	"time"
)

// replayFixture is a recorded run captured for time-faithful playback. Each step
// carries the generation's real results.json blob and the delay (seconds) that
// elapsed before it appeared in the original run, so replay reproduces the run's
// actual cadence rather than dumping every generation at once.
type replayFixture struct {
	Title string       `json:"title"`
	Steps []replayStep `json:"steps"`
}

type replayStep struct {
	Gen     int             `json:"gen"`
	DelayS  float64         `json:"delay_s"`
	Results json.RawMessage `json:"results"`
}

// loadReplay reads and decodes a replay fixture from path.
func loadReplay(path string) (*replayFixture, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var f replayFixture
	if err := json.Unmarshal(data, &f); err != nil {
		return nil, fmt.Errorf("parse replay fixture %s: %w", path, err)
	}
	if len(f.Steps) == 0 {
		return nil, fmt.Errorf("replay fixture %s has no steps", path)
	}
	return &f, nil
}

// gens decodes the fixture's first n steps into a Gen series (run is always 1 in
// a replay). n clamps to the number of steps. The series is the recorded data,
// decoded through the same path as a live results.json — nothing is invented.
func (f *replayFixture) gens(n int) []Gen {
	if n > len(f.Steps) {
		n = len(f.Steps)
	}
	out := make([]Gen, 0, n)
	for i := 0; i < n; i++ {
		s := f.Steps[i]
		g, ok := decodeResult(s.Results, 1, s.Gen)
		if !ok {
			continue
		}
		out = append(out, g)
	}
	sortGens(out)
	return out
}

// delayBefore returns the wall-clock pause to honor before revealing step index i
// (0-based), scaled by speed. speed>1 compresses time (faster demo); the first
// step has no leading delay. A floor keeps the cadence visible even at high speed.
func (f *replayFixture) delayBefore(i int, speed float64) time.Duration {
	if i <= 0 || i >= len(f.Steps) {
		return 0
	}
	if speed <= 0 {
		speed = 1
	}
	d := f.Steps[i].DelayS / speed
	const floor = 0.4 // seconds: keep each reveal perceptible
	if d < floor {
		d = floor
	}
	return time.Duration(d * float64(time.Second))
}

// replayer drives a store through a fixture's steps over time, revealing one
// generation per step with the recorded (speed-scaled) delay between them. It
// plays the run through ONCE and then holds the completed series on screen — a
// demo should freeze on the finished climb, not reset to an empty start. revealed
// is the count of steps currently visible, read under the store's discipline via
// snapshots.
type replayer struct {
	fixture *replayFixture
	speed   float64
	st      *store

	mu       sync.Mutex
	revealed int
}

func newReplayer(f *replayFixture, speed float64, st *store) *replayer {
	return &replayer{fixture: f, speed: speed, st: st}
}

// run plays the fixture forward once, revealing one generation per step, then
// returns — leaving the completed series on screen. It writes the growing series
// into the store and bumps the version so the chart animates each new generation.
// The store's live flag is set true (replay IS real recorded data, faithfully
// timed) but the header tags the source as REPLAY so it is never confused with a
// live run-tree tail; see header().
func (r *replayer) run() {
	for i := range r.fixture.Steps {
		if d := r.fixture.delayBefore(i, r.speed); d > 0 {
			time.Sleep(d)
		}
		r.mu.Lock()
		r.revealed = i + 1
		n := r.revealed
		r.mu.Unlock()

		gens := r.fixture.gens(n)
		r.st.set(gens, true)
		r.st.syncLive()
		r.st.version.Set(r.st.version.Get() + 1)
	}
	// Done: the completed run holds on screen. The heartbeat keeps pulsing so the
	// panel still reads as alive, but the series no longer changes.
}
