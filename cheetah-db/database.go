// database.go
package main

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

type Database struct {
	path            string
	highestKey      atomic.Uint64
	nextPairTableID atomic.Uint32 // Contatore per i nuovi ID delle tabelle pair
	mainKeys        *MainKeysTable
	valuesTables    sync.Map
	recycleTables   sync.Map
	pairTables      sync.Map // Cache per i nodi della TreeTable, ora indicizzata da uint32
	payloadCache    *payloadCache
	mu              sync.Mutex
	pairDir         string // Path alla cartella /pairs
	nextPairIDPath  string // Path al file che memorizza il contatore
	resources       *ResourceMonitor
}

const (
	pairScanDefaultLimit = 256
	pairScanMaxLimit     = 4096
)

type PairScanResult struct {
	Value []byte
	Key   uint64
}

type PairReduceResult struct {
	Value   []byte
	Key     uint64
	Payload []byte
}

const (
	pairEntryKeyOffset   = 1
	pairEntryChildOffset = pairEntryKeyOffset + PairEntryKeySize
)

func entryHasTerminal(entry []byte) bool {
	return len(entry) > 0 && (entry[0]&FlagIsTerminal) != 0
}

func entryHasChild(entry []byte) bool {
	if len(entry) == 0 || (entry[0]&FlagHasChild) == 0 {
		return false
	}
	return binary.BigEndian.Uint32(entry[pairEntryChildOffset:pairEntryChildOffset+PairEntryChildSize]) != 0
}

func entryChildID(entry []byte) uint32 {
	if len(entry) < pairEntryChildOffset+PairEntryChildSize {
		return 0
	}
	return binary.BigEndian.Uint32(entry[pairEntryChildOffset : pairEntryChildOffset+PairEntryChildSize])
}

func setEntryChild(entry []byte, childID uint32) {
	if len(entry) < pairEntryChildOffset+PairEntryChildSize {
		return
	}
	entry[0] |= FlagHasChild
	binary.BigEndian.PutUint32(entry[pairEntryChildOffset:], childID)
}

func clearEntryChild(entry []byte) {
	if len(entry) < pairEntryChildOffset+PairEntryChildSize {
		return
	}
	entry[0] &^= FlagHasChild
	for i := 0; i < PairEntryChildSize; i++ {
		entry[pairEntryChildOffset+i] = 0
	}
}

func setEntryTerminal(entry []byte, absKey uint64) {
	if len(entry) < pairEntryKeyOffset+PairEntryKeySize {
		return
	}
	entry[0] |= FlagIsTerminal
	var buf [8]byte
	binary.BigEndian.PutUint64(buf[:], absKey)
	copy(entry[pairEntryKeyOffset:pairEntryKeyOffset+PairEntryKeySize], buf[8-PairEntryKeySize:])
}

func clearEntryTerminal(entry []byte) {
	if len(entry) < pairEntryKeyOffset+PairEntryKeySize {
		return
	}
	entry[0] &^= FlagIsTerminal
	for i := 0; i < PairEntryKeySize; i++ {
		entry[pairEntryKeyOffset+i] = 0
	}
}

func NewDatabase(path string, monitor *ResourceMonitor) (*Database, error) {
	if err := os.MkdirAll(path, 0755); err != nil {
		return nil, err
	}
	mainKeysPath := filepath.Join(path, "main_keys.table")
	mkt, err := NewMainKeysTable(mainKeysPath)
	if err != nil {
		return nil, err
	}

	pairDir := filepath.Join(path, "pairs")
	if err := os.MkdirAll(pairDir, 0755); err != nil {
		return nil, err
	}

	db := &Database{
		path:           path,
		pairDir:        pairDir,
		nextPairIDPath: filepath.Join(pairDir, "next_id.dat"),
		mainKeys:       mkt,
		payloadCache:   newPayloadCacheFromEnv(),
		resources:      monitor,
	}

	// Carica il contatore degli ID delle tabelle pair
	if err := db.loadNextPairTableID(); err != nil {
		return nil, err
	}

	if err := db.loadHighestKey(); err != nil {
		mkt.Close()
		return nil, err
	}
	return db, nil
}

func (db *Database) Path() string { return db.path }

// Close chiude tutte le tabelle aperte per questo database.
func (db *Database) Close() error {
	var firstErr error
	db.mainKeys.Close()
	db.valuesTables.Range(func(key, value interface{}) bool {
		if table, ok := value.(interface{ Close() }); ok {
			table.Close()
		}
		return true
	})
	db.recycleTables.Range(func(key, value interface{}) bool {
		if table, ok := value.(interface{ Close() }); ok {
			table.Close()
		}
		return true
	})
	return firstErr
}

// getValuesTable e getRecycleTable sono i gestori della cache delle tabelle.
func (db *Database) getValuesTable(size uint32, tableID uint32) (*ValuesTable, error) {
	key := fmt.Sprintf("%d_%d", size, tableID)
	if table, ok := db.valuesTables.Load(key); ok {
		return table.(*ValuesTable), nil
	}
	db.mu.Lock()
	defer db.mu.Unlock()
	if table, ok := db.valuesTables.Load(key); ok {
		return table.(*ValuesTable), nil
	}
	path := filepath.Join(db.path, fmt.Sprintf("values_%s.table", key))
	newTable, err := NewValuesTable(path)
	if err != nil {
		return nil, err
	}
	db.valuesTables.Store(key, newTable)
	return newTable, nil
}

// todo: unificate getValuesTable and getRecycleTable
func (db *Database) getRecycleTable(size uint32) (*RecycleTable, error) {
	key := size
	if table, ok := db.recycleTables.Load(key); ok {
		return table.(*RecycleTable), nil
	}
	db.mu.Lock()
	defer db.mu.Unlock()
	if table, ok := db.recycleTables.Load(key); ok {
		return table.(*RecycleTable), nil
	}
	path := filepath.Join(db.path, fmt.Sprintf("values_%d.recycle.table", size))
	newTable, err := NewRecycleTable(path)
	if err != nil {
		return nil, err
	}
	db.recycleTables.Store(key, newTable)
	return newTable, nil
}

// loadNextPairTableID carica dal disco il prossimo ID da usare per una nuova tabella pair.
func (db *Database) loadNextPairTableID() error {
	data, err := os.ReadFile(db.nextPairIDPath)
	if err != nil {
		if os.IsNotExist(err) {
			// Il file non esiste, partiamo da 1 (0 è la root)
			db.nextPairTableID.Store(1)
			return nil
		}
		return err
	}
	if len(data) >= 4 {
		db.nextPairTableID.Store(binary.BigEndian.Uint32(data))
	}
	return nil
}

// getNewPairTableID restituisce un nuovo ID univoco e lo salva su disco.
func (db *Database) getNewPairTableID() (uint32, error) {
	newID := db.nextPairTableID.Add(1) - 1
	buf := make([]byte, 4)
	binary.BigEndian.PutUint32(buf, newID+1)
	return newID, os.WriteFile(db.nextPairIDPath, buf, 0644)
}

// getPairTable ora accetta un uint32 ID.
func (db *Database) getPairTable(tableID uint32) (*PairTable, error) {
	if table, ok := db.pairTables.Load(tableID); ok {
		return table.(*PairTable), nil
	}
	db.mu.Lock()
	defer db.mu.Unlock()
	if table, ok := db.pairTables.Load(tableID); ok {
		return table.(*PairTable), nil
	}

	// Il nome del file è l'ID in esadecimale
	path := filepath.Join(db.pairDir, fmt.Sprintf("%x.table", tableID))
	newTable, err := NewPairTable(path)
	if err != nil {
		return nil, err
	}
	db.pairTables.Store(tableID, newTable)
	return newTable, nil
}

// ExecuteCommand analizza ed esegue un comando.
func (db *Database) ExecuteCommand(line string) (string, error) {
	trimmed := strings.TrimSpace(line)
	if trimmed == "" {
		return "ERROR,empty_command", nil
	}
	parts := strings.SplitN(trimmed, " ", 2)
	command := strings.ToUpper(parts[0])
	args := ""
	if len(parts) > 1 {
		args = parts[1]
	}

	logVerbosef("Received command=%s args=%s", command, summarizeArg(args))

	var response string
	var err error

	switch {
	case strings.HasPrefix(command, "INSERT"):
		if args == "" {
			response = "ERROR,missing_value"
			break
		}
		value := []byte(args)
		size := 0
		if strings.Contains(command, ":") {
			sizeStr := strings.Split(command, ":")[1]
			size, err = strconv.Atoi(sizeStr)
			if err != nil {
				response = "ERROR,invalid_size_in_command"
				err = nil
				break
			}
		}
		response, err = db.Insert(value, size)
	case command == "READ":
		if args == "" {
			response = "ERROR,missing_key"
			break
		}
		var key uint64
		key, err = strconv.ParseUint(args, 10, 64)
		if err != nil {
			response = "ERROR,invalid_key_format"
			err = nil
			break
		}
		response, err = db.Read(key)
	case command == "EDIT":
		if args == "" {
			response = "ERROR,missing_arguments"
			break
		}
		editArgs := strings.SplitN(args, " ", 2)
		if len(editArgs) < 2 {
			response = "ERROR,edit_requires_key_and_value"
			break
		}
		var key uint64
		key, err = strconv.ParseUint(editArgs[0], 10, 64)
		if err != nil {
			response = "ERROR,invalid_key_format"
			err = nil
			break
		}
		response, err = db.Edit(key, []byte(editArgs[1]))
	case command == "DELETE":
		if args == "" {
			response = "ERROR,missing_key"
			break
		}
		var key uint64
		key, err = strconv.ParseUint(args, 10, 64)
		if err != nil {
			response = "ERROR,invalid_key_format"
			err = nil
			break
		}
		response, err = db.Delete(key)
	case command == "PAIR_SET":
		setArgs := strings.SplitN(args, " ", 2)
		if len(setArgs) < 2 {
			response = "ERROR,pair_set_requires_value_and_key"
			break
		}
		var value []byte
		value, err = parseValue(setArgs[0])
		if err != nil {
			response = err.Error()
			err = nil
			break
		}
		var absKey uint64
		absKey, err = strconv.ParseUint(setArgs[1], 10, 64)
		if err != nil {
			response = "ERROR,invalid_absolute_key_format"
			err = nil
			break
		}
		response, err = db.PairSet(value, absKey)
	case command == "PAIR_GET":
		var value []byte
		value, err = parseValue(args)
		if err != nil {
			response = err.Error()
			err = nil
			break
		}
		response, err = db.PairGet(value)
	case command == "PAIR_DEL":
		var value []byte
		value, err = parseValue(args)
		if err != nil {
			response = err.Error()
			err = nil
			break
		}
		response, err = db.PairDel(value)
	case command == "PAIR_SCAN":
		if args == "" {
			response = "ERROR,pair_scan_requires_prefix"
			break
		}
		fields := strings.Fields(args)
		if len(fields) == 0 {
			response = "ERROR,pair_scan_requires_prefix"
			break
		}
		var prefix []byte
		if fields[0] != "*" {
			prefix, err = parseValue(fields[0])
			if err != nil {
				response = err.Error()
				err = nil
				break
			}
		}
		limit := 0
		if len(fields) > 1 {
			limit, err = strconv.Atoi(fields[1])
			if err != nil {
				response = "ERROR,invalid_limit"
				err = nil
				break
			}
		}
		var cursor []byte
		if len(fields) > 2 {
			if fields[2] != "*" {
				cursor, err = parseValue(fields[2])
				if err != nil {
					response = err.Error()
					err = nil
					break
				}
			}
		}
		var results []PairScanResult
		var nextCursor []byte
		results, nextCursor, err = db.PairScan(prefix, limit, cursor)
		if err != nil {
			response = ""
			break
		}
		response = formatPairScanResponse(results, nextCursor)
	case command == "PAIR_REDUCE":
		if args == "" {
			response = "ERROR,pair_reduce_requires_args"
			break
		}
		fields := strings.Fields(args)
		if len(fields) < 2 {
			response = "ERROR,pair_reduce_requires_mode_and_prefix"
			break
		}
		mode := strings.ToLower(fields[0])
		var prefix []byte
		if fields[1] != "*" {
			prefix, err = parseValue(fields[1])
			if err != nil {
				response = err.Error()
				err = nil
				break
			}
		}
		limit := 0
		if len(fields) > 2 {
			limit, err = strconv.Atoi(fields[2])
			if err != nil {
				response = "ERROR,invalid_limit"
				err = nil
				break
			}
		}
		var cursor []byte
		if len(fields) > 3 {
			if fields[3] != "*" {
				cursor, err = parseValue(fields[3])
				if err != nil {
					response = err.Error()
					err = nil
					break
				}
			}
		}
		response, err = db.handlePairReduce(mode, prefix, limit, cursor)
	case command == "SYSTEM_STATS":
		response = db.systemStatsResponse()
	case command == "LOG_FLUSH":
		limit := 0
		if trimmedArgs := strings.TrimSpace(args); trimmedArgs != "" {
			limit, err = strconv.Atoi(trimmedArgs)
			if err != nil {
				response = "ERROR,invalid_limit"
				err = nil
				break
			}
			if limit < 0 {
				limit = 0
			}
		}
		response = formatLogFlushResponse(logSink.Flush(limit))
	default:
		response = "ERROR,unknown_command"
	}

	if err != nil {
		logErrorf("Command %s failed: %v", command, err)
	} else {
		logVerbosef("Command %s completed -> %s", command, summarizeResponse(response))
	}
	return response, err
}

func (db *Database) deletePairTable(tableID uint32) error {
	// Rimuove dalla cache
	if table, ok := db.pairTables.LoadAndDelete(tableID); ok {
		if pt, castOk := table.(*PairTable); castOk {
			pt.Close() // Chiude il file handle
		}
	}

	// Costruisce il path e rimuove il file dal disco
	path := filepath.Join(db.pairDir, fmt.Sprintf("%x.table", tableID))
	return os.Remove(path)
}

func (db *Database) PairScan(prefix []byte, limit int, cursor []byte) ([]PairScanResult, []byte, error) {
	limit = normalizePairScanLimit(limit)
	results := make([]PairScanResult, 0, limit)
	state := &pairScanState{
		cursor: append([]byte{}, cursor...),
	}
	if len(prefix) == 0 {
		if _, err := db.collectPairEntries(0, nil, limit, &results, state); err != nil {
			return nil, nil, err
		}
		return results, state.nextCursor(), nil
	}

	currentTableID := uint32(0)
	for i, branchByte := range prefix {
		table, err := db.getPairTable(currentTableID)
		if err != nil {
			if os.IsNotExist(err) {
				return results, state.nextCursor(), nil
			}
			return nil, nil, err
		}
		entry, err := table.ReadEntry(branchByte)
		if err != nil {
			return nil, nil, err
		}
		isLast := i == len(prefix)-1
		if isLast {
			if entryHasTerminal(entry) {
				valueCopy := append([]byte{}, prefix...)
				if state.include(valueCopy) {
					results = append(results, PairScanResult{
						Value: valueCopy,
						Key:   decodeAbsoluteKey(entry),
					})
					state.record(valueCopy)
					if limit > 0 && len(results) >= limit {
						state.limitReached = true
						return results, state.nextCursor(), nil
					}
				}
			}
			if !entryHasChild(entry) {
				return results, state.nextCursor(), nil
			}
			childID := entryChildID(entry)
			if childID == 0 {
				return results, state.nextCursor(), nil
			}
			if _, err := db.collectPairEntries(childID, append([]byte{}, prefix...), limit, &results, state); err != nil {
				return nil, nil, err
			}
			return results, state.nextCursor(), nil
		}
		if !entryHasChild(entry) {
			return results, state.nextCursor(), nil
		}
		childID := entryChildID(entry)
		if childID == 0 {
			return results, state.nextCursor(), nil
		}
		currentTableID = childID
	}
	return results, state.nextCursor(), nil
}

func (db *Database) collectPairEntries(tableID uint32, prefix []byte, limit int, acc *[]PairScanResult, state *pairScanState) (bool, error) {
	if limit > 0 && len(*acc) >= limit {
		state.limitReached = true
		return true, nil
	}
	table, err := db.getPairTable(tableID)
	if err != nil {
		if os.IsNotExist(err) {
			return false, nil
		}
		return false, err
	}
	for branch := 0; branch < 256; branch++ {
		entry, err := table.ReadEntry(byte(branch))
		if err != nil {
			return false, err
		}
		if entryHasTerminal(entry) {
			value := append(append([]byte{}, prefix...), byte(branch))
			if state.include(value) {
				*acc = append(*acc, PairScanResult{
					Value: value,
					Key:   decodeAbsoluteKey(entry),
				})
				state.record(value)
				if limit > 0 && len(*acc) >= limit {
					state.limitReached = true
					return true, nil
				}
			}
		}
		if entryHasChild(entry) {
			childID := entryChildID(entry)
			if childID == 0 {
				continue
			}
			nextPrefix := append(append([]byte{}, prefix...), byte(branch))
			reached, err := db.collectPairEntries(childID, nextPrefix, limit, acc, state)
			if err != nil {
				return false, err
			}
			if reached {
				return true, nil
			}
		}
	}
	return false, nil
}

type pairScanState struct {
	cursor          []byte
	cursorSatisfied bool
	lastValue       []byte
	limitReached    bool
}

func (s *pairScanState) include(value []byte) bool {
	if len(s.cursor) == 0 {
		return true
	}
	if s.cursorSatisfied {
		return true
	}
	if bytes.Compare(value, s.cursor) <= 0 {
		return false
	}
	s.cursorSatisfied = true
	return true
}

func (s *pairScanState) record(value []byte) {
	s.lastValue = append([]byte{}, value...)
}

func (s *pairScanState) nextCursor() []byte {
	if !s.limitReached || len(s.lastValue) == 0 {
		return nil
	}
	return append([]byte{}, s.lastValue...)
}

func (db *Database) handlePairReduce(mode string, prefix []byte, limit int, cursor []byte) (string, error) {
	switch mode {
	case "counts", "count", "probabilities", "probs", "backoffs", "continuations", "continuation":
		results, nextCursor, err := db.reduceWithPayload(prefix, limit, cursor)
		if err != nil {
			return "", err
		}
		return formatPairReduceResponse(results, mode, nextCursor), nil
	default:
		return "ERROR,unknown_reducer_mode", nil
	}
}

func (db *Database) reduceWithPayload(prefix []byte, limit int, cursor []byte) ([]PairReduceResult, []byte, error) {
	scanResults, nextCursor, err := db.PairScan(prefix, limit, cursor)
	if err != nil {
		return nil, nil, err
	}
	if len(scanResults) == 0 {
		return nil, nextCursor, nil
	}

	reduced := make([]PairReduceResult, len(scanResults))
	workerCount := len(scanResults)
	if db.resources != nil {
		workerCount = db.resources.RecommendedWorkers(len(scanResults))
	}
	if workerCount == 0 {
		workerCount = runtime.NumCPU() * 2
	}
	if workerCount > len(scanResults) {
		workerCount = len(scanResults)
	}
	if workerCount < 1 {
		workerCount = 1
	}

	sem := make(chan struct{}, workerCount)
	var wg sync.WaitGroup
	var firstErr error
	var errOnce sync.Once
	var abort atomic.Bool

	setErr := func(e error) {
		if e == nil {
			return
		}
		errOnce.Do(func() {
			firstErr = e
			abort.Store(true)
		})
	}

	for idx, res := range scanResults {
		if abort.Load() {
			break
		}
		wg.Add(1)
		go func(i int, res PairScanResult) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()
			if abort.Load() {
				return
			}
			payload, err := db.readValuePayload(res.Key)
			if err != nil {
				setErr(err)
				return
			}
			if abort.Load() {
				return
			}
			reduced[i] = PairReduceResult{
				Value:   res.Value,
				Key:     res.Key,
				Payload: payload,
			}
		}(idx, res)
	}

	wg.Wait()
	if firstErr != nil {
		return nil, nil, firstErr
	}

	return reduced, nextCursor, nil
}

func (db *Database) readValuePayload(key uint64) ([]byte, error) {
	entry, err := db.mainKeys.ReadEntry(key)
	if err != nil {
		return nil, err
	}
	valueSize := readValueSize(entry)
	if valueSize == 0 {
		return nil, fmt.Errorf("key %d has no payload", key)
	}
	location := DecodeValueLocationIndex(entry[ValueSizeBytes:])
	if payload, ok := db.getCachedPayload(valueSize, location); ok {
		return payload, nil
	}
	table, err := db.getValuesTable(valueSize, location.TableID)
	if err != nil {
		return nil, err
	}
	payload := make([]byte, int(valueSize))
	offset := int64(location.EntryID) * int64(valueSize)
	if _, err := table.ReadAt(payload, offset); err != nil {
		return nil, err
	}
	db.cachePayload(valueSize, location, payload)
	return payload, nil
}

func normalizePairScanLimit(limit int) int {
	switch {
	case limit < 0:
		return 0
	case limit == 0:
		return pairScanDefaultLimit
	case limit > pairScanMaxLimit:
		return pairScanMaxLimit
	default:
		return limit
	}
}

func decodeAbsoluteKey(entry []byte) uint64 {
	if len(entry) < pairEntryKeyOffset+PairEntryKeySize {
		return 0
	}
	data := entry[pairEntryKeyOffset : pairEntryKeyOffset+PairEntryKeySize]
	var buf [8]byte
	copy(buf[8-PairEntryKeySize:], data)
	return binary.BigEndian.Uint64(buf[:])
}

func formatPairScanResponse(results []PairScanResult, nextCursor []byte) string {
	var b strings.Builder
	b.WriteString(fmt.Sprintf("SUCCESS,count=%d", len(results)))
	if len(nextCursor) > 0 {
		b.WriteString(fmt.Sprintf(",next_cursor=x%x", nextCursor))
	}
	if len(results) == 0 {
		return b.String()
	}
	b.WriteString(",items=")
	for idx, res := range results {
		if idx > 0 {
			b.WriteString(";")
		}
		b.WriteString(fmt.Sprintf("%x:%d", res.Value, res.Key))
	}
	return b.String()
}

func formatPairReduceResponse(results []PairReduceResult, mode string, nextCursor []byte) string {
	var b strings.Builder
	b.WriteString(fmt.Sprintf("SUCCESS,reducer=%s,count=%d", mode, len(results)))
	if len(nextCursor) > 0 {
		b.WriteString(fmt.Sprintf(",next_cursor=x%x", nextCursor))
	}
	if len(results) == 0 {
		return b.String()
	}
	b.WriteString(",items=")
	for idx, res := range results {
		if idx > 0 {
			b.WriteString(";")
		}
		encoded := base64.StdEncoding.EncodeToString(res.Payload)
		b.WriteString(fmt.Sprintf("%x:%d:%s", res.Value, res.Key, encoded))
	}
	return b.String()
}

func formatLogFlushResponse(entries []string) string {
	if len(entries) == 0 {
		return "SUCCESS,count=0"
	}
	var b strings.Builder
	b.WriteString(fmt.Sprintf("SUCCESS,count=%d", len(entries)))
	for idx, entry := range entries {
		b.WriteString(fmt.Sprintf("\n[%d] %s", idx+1, entry))
	}
	return b.String()
}

func summarizeArg(arg string) string {
	trimmed := strings.TrimSpace(arg)
	if len(trimmed) > 120 {
		return trimmed[:117] + "..."
	}
	return trimmed
}

func summarizeResponse(resp string) string {
	if len(resp) > 160 {
		return resp[:157] + "..."
	}
	return resp
}

func (db *Database) systemStatsResponse() string {
	if db.resources == nil {
		return "ERROR,resource_monitor_unavailable"
	}
	return formatSystemStatsResponse(db.resources.Snapshot())
}

func formatSystemStatsResponse(snap ResourceSnapshot) string {
	var b strings.Builder
	b.WriteString("SUCCESS,command=SYSTEM_STATS")
	if !snap.Timestamp.IsZero() {
		b.WriteString(fmt.Sprintf(",timestamp=%s", snap.Timestamp.UTC().Format(time.RFC3339)))
	}
	b.WriteString(fmt.Sprintf(",logical_cores=%d", snap.LogicalCores))
	b.WriteString(fmt.Sprintf(",gomaxprocs=%d", snap.Gomaxprocs))
	b.WriteString(fmt.Sprintf(",goroutines=%d", snap.Goroutines))
	b.WriteString(fmt.Sprintf(",mem_alloc_bytes=%d", snap.MemAllocBytes))
	b.WriteString(fmt.Sprintf(",mem_sys_bytes=%d", snap.MemSysBytes))
	if snap.ProcessCPUSupported {
		b.WriteString(fmt.Sprintf(",process_cpu_pct=%.2f", snap.ProcessCPUPercent))
	} else {
		b.WriteString(",process_cpu_pct=NA")
	}
	b.WriteString(fmt.Sprintf(",process_cpu_supported=%d", boolToInt(snap.ProcessCPUSupported)))
	if snap.SystemCPUSupported {
		b.WriteString(fmt.Sprintf(",system_cpu_pct=%.2f", snap.SystemCPUPercent))
	} else {
		b.WriteString(",system_cpu_pct=NA")
	}
	b.WriteString(fmt.Sprintf(",system_cpu_supported=%d", boolToInt(snap.SystemCPUSupported)))
	b.WriteString(fmt.Sprintf(",io_supported=%d", boolToInt(snap.IOSupported)))
	if snap.IOSupported {
		b.WriteString(fmt.Sprintf(",io_read_bytes=%d", snap.IOReadBytes))
		b.WriteString(fmt.Sprintf(",io_write_bytes=%d", snap.IOWriteBytes))
		if snap.IOReadRate > 0 {
			b.WriteString(fmt.Sprintf(",io_read_bytes_per_sec=%.2f", snap.IOReadRate))
		} else {
			b.WriteString(",io_read_bytes_per_sec=0")
		}
		if snap.IOWriteRate > 0 {
			b.WriteString(fmt.Sprintf(",io_write_bytes_per_sec=%.2f", snap.IOWriteRate))
		} else {
			b.WriteString(",io_write_bytes_per_sec=0")
		}
	}
	return b.String()
}

func boolToInt(b bool) int {
	if b {
		return 1
	}
	return 0
}

func makePayloadCacheKey(size uint32, location ValueLocationIndex) payloadCacheKey {
	return payloadCacheKey{
		size:    size,
		tableID: location.TableID,
		entryID: location.EntryID,
	}
}

func (db *Database) getCachedPayload(size uint32, location ValueLocationIndex) ([]byte, bool) {
	if db.payloadCache == nil {
		return nil, false
	}
	return db.payloadCache.Get(makePayloadCacheKey(size, location))
}

func (db *Database) cachePayload(size uint32, location ValueLocationIndex, payload []byte) {
	if db.payloadCache == nil || len(payload) == 0 {
		return
	}
	db.payloadCache.Add(makePayloadCacheKey(size, location), payload)
}

func (db *Database) invalidatePayload(size uint32, location ValueLocationIndex) {
	if db.payloadCache == nil {
		return
	}
	db.payloadCache.Invalidate(makePayloadCacheKey(size, location))
}
