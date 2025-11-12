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
)

type Database struct {
	path            string
	highestKey      atomic.Uint64
	nextPairTableID atomic.Uint32 // Contatore per i nuovi ID delle tabelle pair
	mainKeys        *MainKeysTable
	valuesTables    sync.Map
	recycleTables   sync.Map
	pairTables      sync.Map // Cache per i nodi della TreeTable, ora indicizzata da uint32
	mu              sync.Mutex
	pairDir         string // Path alla cartella /pairs
	nextPairIDPath  string // Path al file che memorizza il contatore
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

func NewDatabase(path string) (*Database, error) {
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
	parts := strings.SplitN(line, " ", 2)
	command := strings.ToUpper(parts[0])

	switch {
	case strings.HasPrefix(command, "INSERT"):
		if len(parts) < 2 {
			return "ERROR,missing_value", nil
		}
		value := []byte(parts[1])
		size := 0
		var err error
		if strings.Contains(command, ":") {
			sizeStr := strings.Split(command, ":")[1]
			size, err = strconv.Atoi(sizeStr)
			if err != nil {
				return "ERROR,invalid_size_in_command", nil
			}
		}
		return db.Insert(value, size)
	case command == "READ":
		if len(parts) < 2 {
			return "ERROR,missing_key", nil
		}
		key, err := strconv.ParseUint(parts[1], 10, 64)
		if err != nil {
			return "ERROR,invalid_key_format", nil
		}
		return db.Read(key)
	case command == "EDIT":
		if len(parts) < 2 {
			return "ERROR,missing_arguments", nil
		}
		args := strings.SplitN(parts[1], " ", 2)
		if len(args) < 2 {
			return "ERROR,edit_requires_key_and_value", nil
		}
		key, err := strconv.ParseUint(args[0], 10, 64)
		if err != nil {
			return "ERROR,invalid_key_format", nil
		}
		return db.Edit(key, []byte(args[1]))
	case command == "DELETE":
		if len(parts) < 2 {
			return "ERROR,missing_key", nil
		}
		key, err := strconv.ParseUint(parts[1], 10, 64)
		if err != nil {
			return "ERROR,invalid_key_format", nil
		}
		return db.Delete(key)
	case command == "PAIR_SET":
		args := strings.SplitN(parts[1], " ", 2)
		if len(args) < 2 {
			return "ERROR,pair_set_requires_value_and_key", nil
		}
		value, err := parseValue(args[0])
		if err != nil {
			return err.Error(), nil
		}
		absKey, err := strconv.ParseUint(args[1], 10, 64)
		if err != nil {
			return "ERROR,invalid_absolute_key_format", nil
		}
		return db.PairSet(value, absKey)

	case command == "PAIR_GET":
		value, err := parseValue(parts[1])
		if err != nil {
			return err.Error(), nil
		}
		return db.PairGet(value)

	case command == "PAIR_DEL":
		value, err := parseValue(parts[1])
		if err != nil {
			return err.Error(), nil
		}
		return db.PairDel(value)
	case command == "PAIR_SCAN":
		if len(parts) < 2 {
			return "ERROR,pair_scan_requires_prefix", nil
		}
		args := strings.Fields(parts[1])
		if len(args) == 0 {
			return "ERROR,pair_scan_requires_prefix", nil
		}
		var prefix []byte
		var err error
		if args[0] != "*" {
			prefix, err = parseValue(args[0])
			if err != nil {
				return err.Error(), nil
			}
		}
		limit := 0
		if len(args) > 1 {
			limit, err = strconv.Atoi(args[1])
			if err != nil {
				return "ERROR,invalid_limit", nil
			}
		}
		var cursor []byte
		if len(args) > 2 {
			if args[2] != "*" {
				cursor, err = parseValue(args[2])
				if err != nil {
					return err.Error(), nil
				}
			}
		}
		results, nextCursor, err := db.PairScan(prefix, limit, cursor)
		if err != nil {
			return "", err
		}
		return formatPairScanResponse(results, nextCursor), nil
	case command == "PAIR_REDUCE":
		if len(parts) < 2 {
			return "ERROR,pair_reduce_requires_args", nil
		}
		args := strings.Fields(parts[1])
		if len(args) < 2 {
			return "ERROR,pair_reduce_requires_mode_and_prefix", nil
		}
		mode := strings.ToLower(args[0])
		var prefix []byte
		var err error
		if args[1] != "*" {
			prefix, err = parseValue(args[1])
			if err != nil {
				return err.Error(), nil
			}
		}
		limit := 0
		if len(args) > 2 {
			limit, err = strconv.Atoi(args[2])
			if err != nil {
				return "ERROR,invalid_limit", nil
			}
		}
		var cursor []byte
		if len(args) > 3 {
			if args[3] != "*" {
				cursor, err = parseValue(args[3])
				if err != nil {
					return err.Error(), nil
				}
			}
		}
		response, err := db.handlePairReduce(mode, prefix, limit, cursor)
		if err != nil {
			return "", err
		}
		return response, nil
	default:
		return "ERROR,unknown_command", nil
	}
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
	workerCount := runtime.NumCPU() * 2
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
	table, err := db.getValuesTable(valueSize, location.TableID)
	if err != nil {
		return nil, err
	}
	payload := make([]byte, int(valueSize))
	offset := int64(location.EntryID) * int64(valueSize)
	if _, err := table.ReadAt(payload, offset); err != nil {
		return nil, err
	}
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
