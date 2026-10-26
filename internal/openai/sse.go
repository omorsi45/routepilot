package openai

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"io"
)

// MaxEventBytes is the default cap on the bytes of one SSE event.
const MaxEventBytes = 1 << 20

// ErrEventTooLarge is returned when one event exceeds the reader's size budget.
var ErrEventTooLarge = errors.New("sse: event exceeds size limit")

var doneData = []byte("[DONE]")

// Event is one dispatched server-sent event.
type Event struct {
	Name string
	Data []byte
}

// IsDone reports whether the event is the OpenAI stream terminator.
func (e Event) IsDone() bool {
	return bytes.Equal(e.Data, doneData)
}

// SSEReader parses a server-sent event stream with a hard per-event byte budget.
// Field parsing follows the WHATWG EventSource rules for event and data; id,
// retry, comments and unknown fields are ignored. It is not safe for concurrent use.
type SSEReader struct {
	br  *bufio.Reader
	max int
	err error // sticky terminal error

	line []byte // reusable buffer for the line being assembled
}

// NewSSEReader returns a reader over r. maxEventBytes bounds the bytes consumed
// for a single event; a value <= 0 selects MaxEventBytes.
func NewSSEReader(r io.Reader, maxEventBytes int) *SSEReader {
	if maxEventBytes <= 0 {
		maxEventBytes = MaxEventBytes
	}
	return &SSEReader{br: bufio.NewReader(r), max: maxEventBytes}
}

// Next returns the next event. It returns io.EOF at the end of input; a final
// event that has data but no trailing blank line is returned once first. Read
// errors other than io.EOF are returned as-is, and ErrEventTooLarge is returned
// when the bytes read for one event exceed the budget. Errors are sticky.
func (r *SSEReader) Next() (Event, error) {
	if r.err != nil {
		return Event{}, r.err
	}

	var (
		name    string
		data    []byte
		hasData bool
		used    int
	)
	// dispatchable reports whether the accumulated fields form an event.
	dispatchable := func() bool { return hasData || name != "" }

	for {
		line, err := r.readLine(&used)
		if err != nil && err != io.EOF {
			r.err = err
			return Event{}, err
		}
		atEOF := err == io.EOF

		line = bytes.TrimSuffix(line, []byte("\n"))
		line = bytes.TrimSuffix(line, []byte("\r"))

		if len(line) == 0 && !atEOF {
			if dispatchable() {
				return Event{Name: name, Data: data}, nil
			}
			// Nothing accumulated (comments or stray blank lines): start a fresh budget.
			used = 0
		} else if len(line) > 0 && line[0] != ':' {
			field, value := splitField(line)
			switch field {
			case "event":
				name = string(value)
			case "data":
				if hasData {
					data = append(data, '\n')
				}
				data = append(data, value...)
				hasData = true
			}
		}

		if atEOF {
			// A final event with data but no blank line is emitted once.
			r.err = io.EOF
			if hasData {
				return Event{Name: name, Data: data}, nil
			}
			return Event{}, io.EOF
		}
	}
}

// readLine reads one line including its terminator, adding every byte read to
// *used. It fails with ErrEventTooLarge before buffering past the budget, so
// memory stays within the budget plus one bufio buffer. The returned slice is
// only valid until the next call.
func (r *SSEReader) readLine(used *int) ([]byte, error) {
	r.line = r.line[:0]
	for {
		frag, err := r.br.ReadSlice('\n')
		*used += len(frag)
		if *used > r.max {
			return nil, ErrEventTooLarge
		}
		r.line = append(r.line, frag...)
		if err == bufio.ErrBufferFull {
			continue
		}
		if err != nil && err != io.EOF {
			return nil, err
		}
		return r.line, err
	}
}

// splitField splits "field: value" per EventSource: one optional space after
// the colon is stripped, and a line without a colon is a field with empty value.
func splitField(line []byte) (string, []byte) {
	i := bytes.IndexByte(line, ':')
	if i < 0 {
		return string(line), nil
	}
	value := line[i+1:]
	if len(value) > 0 && value[0] == ' ' {
		value = value[1:]
	}
	return string(line[:i]), value
}

// SSEWriter writes server-sent events to a client and flushes after each one.
type SSEWriter struct {
	w     io.Writer
	flush func() error
}

// NewSSEWriter returns a writer that calls flush (if non-nil) after every event.
func NewSSEWriter(w io.Writer, flush func() error) *SSEWriter {
	return &SSEWriter{w: w, flush: flush}
}

// WriteData writes "data: <b>\n\n" and flushes. b must not contain a line
// break, which would split the payload across events.
func (s *SSEWriter) WriteData(b []byte) error {
	if bytes.ContainsAny(b, "\r\n") {
		return errors.New("sse: data must not contain a line break")
	}
	buf := make([]byte, 0, len(b)+8)
	buf = append(buf, "data: "...)
	buf = append(buf, b...)
	buf = append(buf, '\n', '\n')
	return s.send(buf)
}

// WriteDone writes the "data: [DONE]" terminator and flushes.
func (s *SSEWriter) WriteDone() error {
	return s.send([]byte("data: [DONE]\n\n"))
}

func (s *SSEWriter) send(p []byte) error {
	if _, err := s.w.Write(p); err != nil {
		return fmt.Errorf("sse: write: %w", err)
	}
	if s.flush != nil {
		if err := s.flush(); err != nil {
			return fmt.Errorf("sse: flush: %w", err)
		}
	}
	return nil
}
