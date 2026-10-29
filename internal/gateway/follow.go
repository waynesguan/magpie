package gateway

// A try whose reasoning goes to the agent as it comes (holdWriter.shows)
// may still fail before its reply says anything: its vendor's safety
// filter refuses after the reasoning (#248), or its stream errs. The next
// candidate is asked as before, and its reply follows the reasoning the
// agent has, in the same stream, as one reply (#4166): what begins a reply
// anew (Anthropic's message_start, Responses' response.created and
// response.in_progress) is left out, the blocks and items it opens are
// numbered on from the agent's, what its start counted is told in its end,
// and a Responses reply's end lists the items the agent had before it. A
// Chat or Gemini stream is deltas the agent adds up, which another's
// follow as they would its own (mayReloop).

import (
	"bytes"
	"encoding/json"
	"net/http"
	"slices"
	"strconv"

	"github.com/tidwall/gjson"

	"github.com/yetone/magpie/internal/provider"
)

// follower is what the agent has of a request's reply, shared by its
// tries: once a try's reasoning has gone to it (on), what goes to it is
// read, for the next try's stream to follow.
type follower struct {
	proto provider.Protocol
	// on: a try's reasoning has gone to the agent, and what is written to
	// it from there is read
	on bool
	// begun: the agent has the start of a reply
	begun bool
	// next is the number the agent's next block (Anthropic) or item
	// (Responses) takes; open, those begun and not ended
	next int
	open map[int]bool
	// seq is a Responses stream's next sequence_number; done, the items
	// the agent has whole, by number
	seq  int
	done map[int]json.RawMessage
}

func newFollower(proto provider.Protocol) *follower {
	return &follower{proto: proto, open: map[int]bool{}, done: map[int]json.RawMessage{}}
}

// writer is w for a try about to be asked: once the agent has the start of
// a reply, the try's stream follows it; before, what the try writes goes
// as it is, read from when a try's reasoning shows (on). A nil follower
// is w itself.
func (f *follower) writer(w http.ResponseWriter) http.ResponseWriter {
	if f == nil {
		return w
	}
	fw := &followWriter{ResponseWriter: w, f: f, follows: f.begun, base: f.next, seqBase: f.seq}
	if fw.follows {
		for i := range f.open {
			fw.left = append(fw.left, i)
		}
		slices.Sort(fw.left)
	}
	return fw
}

// followWriter writes a try's stream to the agent, numbered on from what
// the agent has of the reply when it follows one (follows), and reads what
// goes, event by event, while the follower is on.
type followWriter struct {
	http.ResponseWriter
	f *follower
	// follows: the agent has the start of a reply, which this try's
	// stream goes on; base and seqBase are where its numbers begin
	follows bool
	base    int
	seqBase int
	// left are the agent's blocks an earlier try left open, ended before
	// this try's first (stopped)
	left    []int
	stopped bool
	// usage is what the try's start counted, left out with it: told in
	// its message_delta, where the agent reads it too
	usage   []byte
	pending []byte // an event not yet whole
}

func (w *followWriter) Write(b []byte) (int, error) {
	if !w.follows && !w.f.on {
		return w.ResponseWriter.Write(b)
	}
	if !w.follows {
		// nothing to number on: it goes as it is, read for what the
		// agent has
		n, err := w.ResponseWriter.Write(b)
		w.read(b[:n])
		return n, err
	}
	w.pending = append(w.pending, b...)
	var out []byte
	for len(w.pending) > 0 {
		if n := notEvents(w.pending); n > 0 {
			// a comment, a keepalive, goes at once
			out = append(out, w.pending[:n]...)
			w.pending = w.pending[n:]
			continue
		}
		end := eventEnd(w.pending)
		if end < 0 {
			break
		}
		out = append(out, w.event(w.pending[:end])...)
		w.pending = w.pending[end:]
	}
	if len(w.pending) == 0 {
		w.pending = nil
	}
	if len(out) > 0 {
		if _, err := w.ResponseWriter.Write(out); err != nil {
			return 0, err
		}
	}
	return len(b), nil
}

// read notes the whole events of what went to the agent as it was, for
// the next try to follow it.
func (w *followWriter) read(b []byte) {
	w.pending = append(w.pending, b...)
	for len(w.pending) > 0 {
		if n := notEvents(w.pending); n > 0 {
			w.pending = w.pending[n:]
			continue
		}
		end := eventEnd(w.pending)
		if end < 0 {
			break
		}
		w.event(w.pending[:end])
		w.pending = w.pending[end:]
	}
	if len(w.pending) == 0 {
		w.pending = nil
	}
}

func (w *followWriter) Flush() {
	if f, ok := w.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

func (w *followWriter) Unwrap() http.ResponseWriter { return w.ResponseWriter }

// notEvents is how much of b, from its start, is whole lines that are no
// event's: comments and blank lines.
func notEvents(b []byte) int {
	n := 0
	for n < len(b) {
		i := bytes.IndexByte(b[n:], '\n')
		if i < 0 {
			break
		}
		if line := bytes.TrimRight(b[n:n+i], "\r"); len(line) > 0 && line[0] != ':' {
			break
		}
		n += i + 1
	}
	return n
}

// event is one whole event of the try's, as the agent is to have it: read
// for what the agent has, and numbered on when the try follows.
func (w *followWriter) event(ev []byte) []byte {
	_, data := sseParts(ev)
	if len(data) == 0 {
		return ev
	}
	w.f.begun = true
	switch w.f.proto {
	case provider.Anthropic:
		return w.anthropic(ev, data)
	case provider.Responses:
		return w.responses(ev, data)
	}
	return ev
}

func (w *followWriter) anthropic(ev, data []byte) []byte {
	switch typ := gjson.GetBytes(data, "type").String(); typ {
	case "message_start":
		if w.follows {
			if u := gjson.GetBytes(data, "message.usage"); u.IsObject() {
				w.usage = []byte(u.Raw)
			}
			return nil
		}
	case "content_block_start", "content_block_delta", "content_block_stop":
		i := int(gjson.GetBytes(data, "index").Int())
		var out []byte
		if w.follows {
			out = w.stopLeft()
			i += w.base
			ev = eventWithData(ev, setMember(data, "index", []byte(strconv.Itoa(i))))
		}
		switch typ {
		case "content_block_start":
			w.f.open[i] = true
			w.f.next = max(w.f.next, i+1)
		case "content_block_stop":
			delete(w.f.open, i)
		}
		return append(out, ev...)
	case "message_delta":
		if w.follows {
			return append(w.stopLeft(), eventWithData(ev, w.withStartUsage(data))...)
		}
	}
	return ev
}

// stopLeft ends the blocks an earlier try left open, once, before the
// try's own: a reply whose filter stopped it mid-block.
func (w *followWriter) stopLeft() []byte {
	if w.stopped {
		return nil
	}
	w.stopped = true
	var out []byte
	for _, i := range w.left {
		if !w.f.open[i] {
			continue
		}
		delete(w.f.open, i)
		stop, _ := json.Marshal(map[string]any{"type": "content_block_stop", "index": i})
		out = append(out, "event: content_block_stop\ndata: "...)
		out = append(append(out, stop...), "\n\n"...)
	}
	return out
}

// withStartUsage is a message_delta with what the try's message_start
// counted of its prompt and cache where it doesn't say them itself: the
// agent takes them from the delta (Claude Code and the SDKs keep the
// start's otherwise, which were another try's).
func (w *followWriter) withStartUsage(data []byte) []byte {
	if len(w.usage) == 0 {
		return data
	}
	usage := gjson.GetBytes(data, "usage")
	out := []byte(usage.Raw)
	if !usage.IsObject() {
		out = []byte("{}")
	}
	changed := false
	for _, k := range []string{"input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "cache_creation"} {
		v := gjson.GetBytes(w.usage, k)
		if !v.Exists() || v.Type == gjson.Null {
			continue
		}
		if d := gjson.Get(string(out), k); d.Exists() && d.Type != gjson.Null {
			continue
		}
		out, changed = setMember(out, k, []byte(v.Raw)), true
	}
	if !changed {
		return data
	}
	return setMember(data, "usage", out)
}

func (w *followWriter) responses(ev, data []byte) []byte {
	typ := gjson.GetBytes(data, "type").String()
	if w.follows {
		switch typ {
		case "response.created", "response.in_progress", "response.queued":
			return nil
		}
		changed := false
		if s := gjson.GetBytes(data, "sequence_number"); s.Exists() {
			data, changed = setMember(data, "sequence_number", []byte(strconv.FormatInt(int64(w.seqBase)+s.Int(), 10))), true
		}
		if o := gjson.GetBytes(data, "output_index"); o.Exists() {
			data, changed = setMember(data, "output_index", []byte(strconv.FormatInt(int64(w.base)+o.Int(), 10))), true
		}
		switch typ {
		case "response.completed", "response.incomplete", "response.failed", "response.done":
			if b, ok := w.withItemsBefore(data); ok {
				data, changed = b, true
			}
		}
		if changed {
			ev = eventWithData(ev, data)
		}
	}
	if s := gjson.GetBytes(data, "sequence_number"); s.Exists() {
		w.f.seq = max(w.f.seq, int(s.Int())+1)
	}
	if o := gjson.GetBytes(data, "output_index"); o.Exists() {
		i := int(o.Int())
		w.f.next = max(w.f.next, i+1)
		if item := gjson.GetBytes(data, "item"); typ == "response.output_item.done" && item.IsObject() {
			w.f.done[i] = json.RawMessage(item.Raw)
		}
	}
	return ev
}

// withItemsBefore is a Responses reply's end with the items the agent had
// whole before the try first in its output, as it had them streamed: a
// client that keeps the response its end gives (the SDKs' final
// response) has the whole reply.
func (w *followWriter) withItemsBefore(data []byte) ([]byte, bool) {
	resp := gjson.GetBytes(data, "response")
	if !resp.IsObject() {
		return data, false
	}
	var before []int
	for i := range w.f.done {
		if i < w.base {
			before = append(before, i)
		}
	}
	if len(before) == 0 {
		return data, false
	}
	slices.Sort(before)
	items := make([]json.RawMessage, 0, len(before))
	for _, i := range before {
		items = append(items, w.f.done[i])
	}
	if out := resp.Get("output"); out.IsArray() {
		for _, el := range out.Array() {
			items = append(items, json.RawMessage(el.Raw))
		}
	}
	list, err := json.Marshal(items)
	if err != nil {
		return data, false
	}
	return setMember(data, "response", setMember([]byte(resp.Raw), "output", list)), true
}

// sseParts is an event's name and its data, its data lines joined as
// streamEvent reads them.
func sseParts(ev []byte) (name string, data []byte) {
	for _, ln := range bytes.Split(ev, []byte("\n")) {
		ln = bytes.TrimRight(ln, "\r")
		switch {
		case bytes.HasPrefix(ln, []byte("event:")):
			name = string(bytes.TrimSpace(ln[6:]))
		case bytes.HasPrefix(ln, []byte("data:")):
			data = append(data, bytes.TrimSpace(ln[5:])...)
		}
	}
	return name, data
}

// eventWithData is an event with data in place of its own: its other lines
// (its name, an id, a comment) as they were, its data on one line where
// its first was.
func eventWithData(ev, data []byte) []byte {
	var out []byte
	put := false
	for _, ln := range bytes.Split(bytes.TrimRight(ev, "\r\n"), []byte("\n")) {
		if bytes.HasPrefix(bytes.TrimRight(ln, "\r"), []byte("data:")) {
			if !put {
				out = append(append(append(out, "data: "...), data...), '\n')
				put = true
			}
			continue
		}
		out = append(append(out, bytes.TrimRight(ln, "\r")...), '\n')
	}
	if !put {
		out = append(append(append(out, "data: "...), data...), '\n')
	}
	return append(out, '\n')
}

// setMember is obj, a JSON object, with its member key's value val: in
// place where it has one, every other byte as it came, else added at its
// end. What isn't an object is returned as it is.
func setMember(obj []byte, key string, val []byte) []byte {
	obj = trimJSON(obj)
	if len(obj) < 2 || obj[0] != '{' || !jsonCheck(obj) {
		return obj
	}
	at, end := -1, -1
	eachMemberAt(obj, func(k0, k1, v0, v1 int) bool {
		if keyIs(obj[k0:k1], key) {
			at, end = v0, v1
			return false
		}
		return true
	})
	out := make([]byte, 0, len(obj)+len(key)+len(val)+4)
	if at >= 0 {
		out = append(out, obj[:at]...)
		out = append(out, val...)
		return append(out, obj[end:]...)
	}
	k, _ := json.Marshal(key)
	out = append(out, obj[:len(obj)-1]...)
	if len(bytes.TrimSpace(obj[1:len(obj)-1])) > 0 {
		out = append(out, ',')
	}
	out = append(append(append(out, k...), ':'), val...)
	return append(out, '}')
}
