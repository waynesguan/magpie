package gateway

import (
	"bufio"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/tidwall/gjson"

	"github.com/yetone/magpie/internal/provider"
)

// sseEvents is a stream's events, by name and data, in order: what an
// agent reads of it.
func sseEvents(body string) (names []string, datas []string) {
	for _, ev := range strings.Split(body, "\n\n") {
		name, data := sseParts([]byte(ev))
		if name == "" && len(data) == 0 {
			continue
		}
		names, datas = append(names, name), append(datas, string(data))
	}
	return names, datas
}

// anthropicWhole reads an Anthropic stream as the SDKs' MessageStream does:
// one message_start before anything, each block begun once, at the next
// index, and stopped, no event after message_stop. It says what is wrong,
// or the message's blocks by type and the usage its end left it with.
func anthropicWhole(body string) (blocks []string, usage map[string]int64, wrong string) {
	_, datas := sseEvents(body)
	started, stopped := false, false
	open := map[int64]bool{}
	usage = map[string]int64{}
	for _, d := range datas {
		typ := gjson.Get(d, "type").String()
		if stopped {
			return nil, nil, "an event after message_stop: " + d
		}
		if typ != "message_start" && typ != "ping" && typ != "error" && !started {
			return nil, nil, "an event before message_start: " + d
		}
		switch typ {
		case "message_start":
			if started {
				return nil, nil, "a second message_start: " + d
			}
			started = true
			gjson.Get(d, "message.usage").ForEach(func(k, v gjson.Result) bool { usage[k.String()] = v.Int(); return true })
		case "content_block_start":
			i := gjson.Get(d, "index").Int()
			if i != int64(len(blocks)) {
				return nil, nil, "a block begun out of order: " + d
			}
			blocks = append(blocks, gjson.Get(d, "content_block.type").String())
			open[i] = true
		case "content_block_delta":
			if !open[gjson.Get(d, "index").Int()] {
				return nil, nil, "a delta to no open block: " + d
			}
		case "content_block_stop":
			i := gjson.Get(d, "index").Int()
			if !open[i] {
				return nil, nil, "a stop of no open block: " + d
			}
			delete(open, i)
		case "message_delta":
			gjson.Get(d, "usage").ForEach(func(k, v gjson.Result) bool { usage[k.String()] = v.Int(); return true })
		case "message_stop":
			stopped = true
		}
	}
	if len(open) > 0 || !stopped {
		return nil, nil, "a block left open, or no message_stop"
	}
	return blocks, usage, ""
}

// Claude thinking then refused, with an empty text block begun between, as
// Anthropic stops a reply its filter caught at the text's start; what
// another said after it.
var (
	anthropicThoughtFramedRefused = sse(`event: message_start`+"\n"+`data: {"type":"message_start","message":{"id":"m1","role":"assistant","model":"m","content":[],"usage":{"input_tokens":2,"cache_read_input_tokens":32749}}}`,
		`event: content_block_start`+"\n"+`data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}`,
		`event: content_block_delta`+"\n"+`data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"The user wants a plan"}}`,
		`event: content_block_delta`+"\n"+`data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig"}}`,
		`event: content_block_stop`+"\n"+`data: {"type":"content_block_stop","index":0}`,
		`event: content_block_start`+"\n"+`data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}`,
		`event: content_block_stop`+"\n"+`data: {"type":"content_block_stop","index":1}`,
		`event: message_delta`+"\n"+`data: {"type":"message_delta","delta":{"stop_reason":"refusal","stop_sequence":null},"usage":{"output_tokens":381}}`,
		`event: message_stop`+"\n"+`data: {"type":"message_stop"}`)
	anthropicThoughtAnswered = sse(`event: message_start`+"\n"+`data: {"type":"message_start","message":{"id":"m2","role":"assistant","model":"m","content":[],"usage":{"input_tokens":5,"cache_creation_input_tokens":40000}}}`,
		`event: content_block_start`+"\n"+`data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}`,
		`event: content_block_delta`+"\n"+`data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"b thinks"}}`,
		`event: content_block_delta`+"\n"+`data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig-b"}}`,
		`event: content_block_stop`+"\n"+`data: {"type":"content_block_stop","index":0}`,
		`event: content_block_start`+"\n"+`data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}`,
		`event: content_block_delta`+"\n"+`data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"from b"}}`,
		`event: content_block_stop`+"\n"+`data: {"type":"content_block_stop","index":1}`,
		`event: message_delta`+"\n"+`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":7}}`,
		`event: message_stop`+"\n"+`data: {"type":"message_stop"}`)
)

// claudeAt saves provider id serving Claude, by its own Messages API.
func claudeAt(t *testing.T, id string, s *scripted) {
	t.Helper()
	up := httptest.NewServer(s)
	t.Cleanup(up.Close)
	if err := provider.Save(provider.Provider{ID: id, Name: strings.ToUpper(id), Key: "k", Models: []string{"claude-opus-5-5"}, Anthropic: up.URL}); err != nil {
		t.Fatal(err)
	}
}

// A group's Claude refusing after its reasoning went to Claude Code, with
// an empty text block begun: the next member's reply follows the reasoning
// as one message the SDKs read whole, its blocks numbered on, its own
// start's usage in its end, and none of the refused try's text block (an
// empty one, which Anthropic would turn away when it is handed back) nor
// its refusal (#4166).
func TestRefusedReasoningIsFollowed(t *testing.T) {
	fresh(t)
	a := &scripted{replies: []reply{{200, "text/event-stream", anthropicThoughtFramedRefused}}}
	b := &scripted{replies: []reply{{200, "text/event-stream", anthropicThoughtAnswered}}}
	claudeAt(t, "a", a)
	claudeAt(t, "b", b)
	refusalGroup(t, "a/claude-opus-5-5", "b/claude-opus-5-5")
	s := New()
	code, body := sendTo(s, "/v1/messages", claudeCodeAsk)
	blocks, usage, wrong := anthropicWhole(body)
	if code != 200 || wrong != "" || a.n != 1 || b.n != 1 {
		t.Fatalf("%d %s: %s (a %d, b %d)", code, wrong, body, a.n, b.n)
	}
	if strings.Join(blocks, ",") != "thinking,thinking,text" || !strings.Contains(body, "The user wants a plan") || !strings.Contains(body, "from b") ||
		strings.Contains(body, `"refusal"`) {
		t.Fatalf("blocks %v: %s", blocks, body)
	}
	// the answer's own prompt and cache, not the refused try's
	if usage["input_tokens"] != 5 || usage["cache_creation_input_tokens"] != 40000 || usage["output_tokens"] != 7 {
		t.Fatalf("usage %v: %s", usage, body)
	}
	r := lastRoute(s)
	if len(r.Tries) != 2 || r.Tries[0].Fail != failRefused || r.Tries[0].Rest != nil || r.Status != 200 {
		t.Fatalf("tries: %+v", r.Tries)
	}
}

// Every member refusing after its reasoning: the last one's refusal
// reaches Claude Code as Claude's own, after the reasoning of each, in one
// message, as it does when nobody else is left (iTianbao on X).
func TestEveryMemberRefusesAfterReasoning(t *testing.T) {
	fresh(t)
	a := &scripted{replies: []reply{{200, "text/event-stream", anthropicThoughtThenRefused}}}
	b := &scripted{replies: []reply{{200, "text/event-stream", anthropicThoughtThenRefused}}}
	claudeAt(t, "a", a)
	claudeAt(t, "b", b)
	refusalGroup(t, "a/claude-opus-5-5", "b/claude-opus-5-5")
	code, body := sendTo(New(), "/v1/messages", claudeCodeAsk)
	blocks, _, wrong := anthropicWhole(body)
	if code != 200 || wrong != "" || strings.Join(blocks, ",") != "thinking,thinking" || !strings.Contains(body, `"stop_reason":"refusal"`) || a.n != 1 || b.n != 1 {
		t.Fatalf("%d %s %v: %s (a %d, b %d)", code, wrong, blocks, body, a.n, b.n)
	}
}

// With Settings' RefusalRetries at 0 nobody else is asked after a
// refusal: the reasoning streams as on the last candidate, and the
// refusal after it goes to Claude Code as Claude's own.
func TestReasoningStreamsWhenRefusalsGoBack(t *testing.T) {
	fresh(t)
	refusalRetries(t, 0)
	a := &scripted{replies: []reply{{200, "text/event-stream", anthropicThoughtThenRefused}}}
	b := &scripted{replies: []reply{{200, "text/event-stream", anthropicAnswer}}}
	claudeAt(t, "a", a)
	claudeAt(t, "b", b)
	refusalGroup(t, "a/claude-opus-5-5", "b/claude-opus-5-5")
	code, body := sendTo(New(), "/v1/messages", claudeCodeAsk)
	if _, _, wrong := anthropicWhole(body); code != 200 || wrong != "" || !strings.Contains(body, `"stop_reason":"refusal"`) || a.n != 1 || b.n != 0 {
		t.Fatalf("%d %s: %s (a %d, b %d)", code, wrong, body, a.n, b.n)
	}
}

// An error after a group's Claude reasoned, nothing said, goes to the next
// member as one before anything was said did while the reasoning was held
// (#248): the answer follows the reasoning the agent has.
func TestErrorAfterReasoningIsFollowed(t *testing.T) {
	overloaded := strings.Replace(anthropicThoughtThenRefused,
		`event: message_delta`+"\n"+`data: {"type":"message_delta","delta":{"stop_reason":"refusal","stop_sequence":null},"usage":{"output_tokens":381}}`,
		`event: error`+"\n"+`data: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`, 1)
	fresh(t)
	a := &scripted{replies: []reply{{200, "text/event-stream", overloaded}}}
	b := &scripted{replies: []reply{{200, "text/event-stream", anthropicAnswer}}}
	claudeAt(t, "a", a)
	claudeAt(t, "b", b)
	refusalGroup(t, "a/claude-opus-5-5", "b/claude-opus-5-5")
	s := New()
	code, body := sendTo(s, "/v1/messages", claudeCodeAsk)
	blocks, _, wrong := anthropicWhole(body)
	if code != 200 || wrong != "" || strings.Join(blocks, ",") != "thinking,text" || strings.Contains(body, "Overloaded") || a.n != 1 || b.n != 1 {
		t.Fatalf("%d %s %v: %s (a %d, b %d)", code, wrong, blocks, body, a.n, b.n)
	}
	if r := lastRoute(s); len(r.Tries) != 2 || r.Status != 200 {
		t.Fatalf("tries: %+v", r.Tries)
	}
}

// Codex on a group whose first member's Claude refused after reasoning:
// one response, the next member's items numbered on from the reasoning
// item Codex has, their events' sequence going on, and the response's end
// listing every item, the reasoning first.
func TestRefusedReasoningIsFollowedOnResponses(t *testing.T) {
	fresh(t)
	a := &scripted{replies: []reply{{200, "text/event-stream", anthropicThoughtThenRefused}}}
	b := &scripted{replies: []reply{{200, "text/event-stream", anthropicAnswer}}}
	claudeAt(t, "a", a)
	claudeAt(t, "b", b)
	refusalGroup(t, "a/claude-opus-5-5", "b/claude-opus-5-5")
	code, body := sendTo(New(), "/v1/responses", codexAsk)
	names, datas := sseEvents(body)
	if code != 200 || a.n != 1 || b.n != 1 || strings.Count(body, "event: response.created") != 1 || strings.Contains(body, "refusal") {
		t.Fatalf("%d: %s (a %d, b %d)", code, body, a.n, b.n)
	}
	seq, added := int64(-1), map[int64]string{}
	var end string
	for i, d := range datas {
		if s := gjson.Get(d, "sequence_number"); s.Exists() {
			if s.Int() <= seq {
				t.Fatalf("sequence %d after %d: %s", s.Int(), seq, body)
			}
			seq = s.Int()
		}
		if names[i] == "response.output_item.added" {
			o := gjson.Get(d, "output_index").Int()
			if _, twice := added[o]; twice {
				t.Fatalf("item %d added twice: %s", o, body)
			}
			added[o] = gjson.Get(d, "item.type").String()
		}
		if names[i] == "response.completed" {
			end = d
		}
	}
	if added[0] != "reasoning" || added[1] != "message" || len(added) != 2 {
		t.Fatalf("items %v: %s", added, body)
	}
	var out []string
	for _, it := range gjson.Get(end, "response.output").Array() {
		out = append(out, it.Get("type").String())
	}
	if strings.Join(out, ",") != "reasoning,message" || !strings.Contains(end, "from b") {
		t.Fatalf("the end lists %v: %s", out, end)
	}
}

// A Chat agent and a Gemini one on a group whose first member's Claude
// refused after reasoning: their chunks are deltas they add up, and the
// next member's follow the reasoning's as they would its own.
func TestRefusedReasoningIsFollowedOnChatAndGemini(t *testing.T) {
	for _, x := range []struct{ name, path, ask, refusal string }{
		{"chat", "/v1/chat/completions", `{"model":"group/g","stream":true,"messages":[{"role":"user","content":"hi"}]}`, "content_filter"},
		{"gemini", "/v1beta/models/group/g:streamGenerateContent?alt=sse", `{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"generationConfig":{"thinkingConfig":{"includeThoughts":true}}}`, "SAFETY"},
	} {
		t.Run(x.name, func(t *testing.T) {
			fresh(t)
			a := &scripted{replies: []reply{{200, "text/event-stream", anthropicThoughtThenRefused}}}
			b := &scripted{replies: []reply{{200, "text/event-stream", anthropicAnswer}}}
			claudeAt(t, "a", a)
			claudeAt(t, "b", b)
			refusalGroup(t, "a/claude-opus-5-5", "b/claude-opus-5-5")
			code, body := sendTo(New(), x.path, x.ask)
			thought, answer := strings.Index(body, "user wants"), strings.Index(body, "from b")
			if code != 200 || thought < 0 || answer < thought || strings.Contains(body, x.refusal) || strings.Contains(body, `"refusal"`) || a.n != 1 || b.n != 1 {
				t.Fatalf("%d: %s (a %d, b %d)", code, body, a.n, b.n)
			}
		})
	}
}

// A group's Claude reasoning a while before it answers: its reasoning
// reaches Claude Code and Codex as it comes, through a group as on its
// own, while a refusal after it could still go to the next member (#4166).
// Held for that, Claude Code had the first of it only with the text, and
// so never showed "thinking with <effort> effort".
func TestGroupReasoningStreams(t *testing.T) {
	claudeThinks := make([]string, len(glmThinksSlowly))
	for i, ev := range glmThinksSlowly {
		claudeThinks[i] = strings.Replace(ev, `"glm-5.1"`, `"claude-opus-5-5"`, 1)
	}
	for _, ask := range []struct{ path, body, thought, text string }{
		{"/v1/messages", claudeCodeAsk, `"thinking_delta"`, `"text_delta"`},
		{"/v1/responses", codexAsk, `"response.reasoning_summary_text.delta"`, `"response.output_text.delta"`},
	} {
		t.Run(ask.path, func(t *testing.T) {
			fresh(t)
			up := &slowThinker{events: claudeThinks, gap: 200 * time.Millisecond, done: make(chan time.Time, 1)}
			vendor := httptest.NewServer(up)
			t.Cleanup(vendor.Close)
			if err := provider.Save(provider.Provider{ID: "a", Name: "A", Key: "k", Models: []string{"claude-opus-5-5"}, Anthropic: vendor.URL}); err != nil {
				t.Fatal(err)
			}
			claudeAt(t, "b", &scripted{replies: []reply{{200, "text/event-stream", anthropicAnswer}}})
			refusalGroup(t, "a/claude-opus-5-5", "b/claude-opus-5-5")
			gw := httptest.NewServer(New().Handler())
			t.Cleanup(gw.Close)
			res, err := http.Post(gw.URL+ask.path, "application/json", strings.NewReader(ask.body))
			if err != nil {
				t.Fatal(err)
			}
			defer res.Body.Close()
			var firstThought, firstText time.Time
			rd := bufio.NewReader(res.Body)
			for {
				ln, err := rd.ReadString('\n')
				if firstThought.IsZero() && strings.Contains(ln, ask.thought) {
					firstThought = time.Now()
				}
				if firstText.IsZero() && strings.Contains(ln, ask.text) {
					firstText = time.Now()
				}
				if err != nil {
					break
				}
			}
			select {
			case <-up.done:
			case <-time.After(5 * time.Second):
				t.Fatal("the vendor never finished")
			}
			if firstThought.IsZero() || firstText.IsZero() {
				t.Fatal("no thinking or no text reached the client")
			}
			// three thoughts 200ms apart come before the text
			if gap := firstText.Sub(firstThought); gap < 300*time.Millisecond {
				t.Fatalf("thinking held: first thought only %v before the text", gap)
			}
		})
	}
}

// What the follower writes of a try's stream: a try that follows nothing
// goes as it is, half an event too, read for what the agent has; one that
// follows sends a comment at once and an event only once whole, keeps a
// message_delta's own usage over its start's, and ends a block an earlier
// try left open before its first.
func TestFollowWriter(t *testing.T) {
	f := newFollower(provider.Anthropic)
	f.on = true
	rec := httptest.NewRecorder()
	w := f.writer(rec)
	first := "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":2}}}\n\nevent: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\"}}\n\nevent: content_block_delta\ndata: {\"type\":\"con"
	w.Write([]byte(first))
	if rec.Body.String() != first || !f.begun || f.next != 1 || !f.open[0] {
		t.Fatalf("%q next %d open %v", rec.Body.String(), f.next, f.open)
	}
	w.Write([]byte("tent_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"hm\"}}\n\n"))
	if !strings.Contains(rec.Body.String(), `"thinking":"hm"}}`) || f.next != 1 || !f.open[0] {
		t.Fatalf("%q next %d open %v", rec.Body.String(), f.next, f.open)
	}

	// the next try, the first's block left open
	rec = httptest.NewRecorder()
	w = f.writer(rec)
	w.Write([]byte(": keepalive\n"))
	if rec.Body.String() != ": keepalive\n" {
		t.Fatalf("the comment waited: %q", rec.Body.String())
	}
	w.Write([]byte("event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":5,\"cache_read_input_tokens\":9}}}\n\n" +
		"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_bl"))
	if strings.Contains(rec.Body.String(), "content_block") || strings.Contains(rec.Body.String(), "message_start") {
		t.Fatalf("half an event, or the start, went: %q", rec.Body.String())
	}
	w.Write([]byte("ock\":{\"type\":\"text\",\"text\":\"\"}}\n\n" +
		"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":3,\"input_tokens\":6}}\n\n"))
	_, datas := sseEvents(rec.Body.String())
	if len(datas) != 3 || gjson.Get(datas[0], "type").String() != "content_block_stop" || gjson.Get(datas[0], "index").Int() != 0 ||
		gjson.Get(datas[1], "index").Int() != 1 {
		t.Fatalf("%q", rec.Body.String())
	}
	var u map[string]int
	if json.Unmarshal([]byte(gjson.Get(datas[2], "usage").Raw), &u) != nil || u["input_tokens"] != 6 || u["cache_read_input_tokens"] != 9 || u["output_tokens"] != 3 {
		t.Fatalf("usage %v: %s", u, datas[2])
	}
}
