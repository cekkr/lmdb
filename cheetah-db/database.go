// database.go
package main

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
)

type Database struct {
	name            string
	path            string
	highestKey      atomic.Uint64
	nextPairTableID atomic.Uint32 // Contatore per i nuovi ID delle tabelle pair
	mainKeys        *MainKeysTable
	valuesTables    sync.Map
	recycleTables   sync.Map
	pairTables      sync.Map // Cache per i nodi della TreeTable, ora indicizzata da uint32
	fileManager     *FileManager
	payloadCache    *payloadCache
	mu              sync.Mutex
	pairDir         string // Path alla cartella /pairs
	nextPairIDPath  string // Path al file che memorizza il contatore
	resources       *ResourceMonitor
}

func resolvePairTableLimit() int {
	if raw := strings.TrimSpace(os.Getenv("CHEETAH_MAX_PAIR_TABLES")); raw != "" {
		if val, err := strconv.Atoi(raw); err == nil && val > 0 {
			return val
		}
	}
	limit := fileDescriptorSoftLimit()
	if limit > 0 {
		candidate := limit - pairTableSafetyMargin
		if candidate < minPairTableLimit {
			candidate = minPairTableLimit
		}
		return candidate
	}
	return defaultPairTableLimit
}

func fileDescriptorSoftLimit() int {
	var rl syscall.Rlimit
	if err := syscall.Getrlimit(syscall.RLIMIT_NOFILE, &rl); err != nil {
		return 0
	}
	if rl.Cur <= 0 || rl.Cur > math.MaxInt32 {
		return 0
	}
	return int(rl.Cur)
}

const (
	pairScanDefaultLimit          = 256
	pairScanMaxLimit              = 4096
	pairSummaryDefaultDepth       = 1
	pairSummaryDefaultBranchLimit = 32
	pairSummaryMaxBranchLimit     = 1024
	defaultPairTableLimit         = 1024
	minPairTableLimit             = 64
	pairTableSafetyMargin         = 128
)

type PairScanResult struct {
	Value []byte
	Key   uint64
}

type pairSummaryBranch struct {
	Path  []byte
	Count int64
}

type PairSummaryResult struct {
	Prefix            []byte
	TerminalCount     int64
	TotalPayloadBytes int64
	MinPayloadBytes   uint32
	MaxPayloadBytes   uint32
	MinKey            uint64
	MaxKey            uint64
	MaxDepth          int
	SelfTerminal      bool
	Branches          []pairSummaryBranch
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

func NewDatabase(name, path string, monitor *ResourceMonitor) (*Database, error) {
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

	fileManager := NewFileManager(resolvePairTableLimit())
	db := &Database{
		name:           name,
		path:           path,
		pairDir:        pairDir,
		nextPairIDPath: filepath.Join(pairDir, "next_id.dat"),
		mainKeys:       mkt,
		payloadCache:   newPayloadCacheFromEnv(),
		resources:      monitor,
		fileManager:    fileManager,
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
func (db *Database) Name() string { return db.name }

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
	db.pairTables.Range(func(key, value interface{}) bool {
		if table, ok := value.(interface{ Close() }); ok {
			table.Close()
		}
		return true
	})
	if db.fileManager != nil {
		db.fileManager.Close()
	}
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
	if table, ok := db.loadPairTable(tableID); ok {
		return table, nil
	}
	db.mu.Lock()
	defer db.mu.Unlock()
	if table, ok := db.loadPairTable(tableID); ok {
		return table, nil
	}

	// Il nome del file è l'ID in esadecimale
	path := filepath.Join(db.pairDir, fmt.Sprintf("%x.table", tableID))
	newTable, err := NewPairTable(db.fileManager, tableID, path)
	if err != nil {
		return nil, err
	}
	db.storePairTable(tableID, newTable)
	return newTable, nil
}

func (db *Database) loadPairTable(tableID uint32) (*PairTable, bool) {
	if table, ok := db.pairTables.Load(tableID); ok {
		if pt, castOk := table.(*PairTable); castOk {
			return pt, true
		}
	}
	return nil, false
}

func (db *Database) storePairTable(tableID uint32, table *PairTable) {
	db.pairTables.Store(tableID, table)
}

func (db *Database) closePairTable(tableID uint32, table *PairTable, deleteFile bool) error {
	if table != nil {
		table.Close()
	}
	if deleteFile {
		path := filepath.Join(db.pairDir, fmt.Sprintf("%x.table", tableID))
		return os.Remove(path)
	}
	return nil
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
	case command == "PAIR_PURGE":
		if args == "" {
			response = "ERROR,pair_purge_requires_prefix"
			break
		}
		fields := strings.Fields(args)
		if len(fields) == 0 {
			response = "ERROR,pair_purge_requires_prefix"
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
		var removed int
		removed, err = db.PairPurge(prefix, limit)
		if err != nil {
			response = ""
			break
		}
		response = fmt.Sprintf("SUCCESS,purged=%d", removed)
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
	case command == "PAIR_SUMMARY":
		if args == "" {
			response = "ERROR,pair_summary_requires_prefix"
			break
		}
		fields := strings.Fields(args)
		if len(fields) == 0 {
			response = "ERROR,pair_summary_requires_prefix"
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
		depth := pairSummaryDefaultDepth
		if len(fields) > 1 {
			depth, err = strconv.Atoi(fields[1])
			if err != nil {
				response = "ERROR,invalid_depth"
				err = nil
				break
			}
		}
		branchLimit := pairSummaryDefaultBranchLimit
		if len(fields) > 2 {
			branchLimit, err = strconv.Atoi(fields[2])
			if err != nil {
				response = "ERROR,invalid_branch_limit"
				err = nil
				break
			}
		}
		var summary *PairSummaryResult
		summary, err = db.PairSummary(prefix, depth, branchLimit)
		if err != nil {
			response = ""
			break
		}
		response = formatPairSummaryResponse(summary)
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
	case command == "FILE_CHECKPOINT":
		if db.fileManager == nil {
			response = "ERROR,file_manager_unavailable"
			break
		}
		var cpOpts FileCheckpointOptions
		cpOpts, err = parseFileCheckpointArgs(args)
		if err != nil {
			response = fmt.Sprintf("ERROR,%v", err)
			err = nil
			break
		}
		count := db.fileManager.ForceCheckpoint(cpOpts)
		response = fmt.Sprintf("SUCCESS,file_checkpoint_flushed=%d", count)
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
	var pt *PairTable
	if table, ok := db.pairTables.LoadAndDelete(tableID); ok {
		if cast, castOk := table.(*PairTable); castOk {
			pt = cast
		}
	}
	return db.closePairTable(tableID, pt, true)
}

func (db *Database) PairScan(prefix []byte, limit int, cursor []byte) ([]PairScanResult, []byte, error) {
	limit = normalizePairScanLimit(limit)
	acc := newPairScanAccumulator(limit, cursor)
	startTable := uint32(0)
	if len(prefix) > 0 {
		currentTableID := uint32(0)
		for i, branchByte := range prefix {
			table, err := db.getPairTable(currentTableID)
			if err != nil {
				if os.IsNotExist(err) {
					return []PairScanResult{}, nil, nil
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
					if acc.add(prefix, decodeAbsoluteKey(entry)) && limit > 0 {
						results, nextCursor := acc.finalize(acc.limit)
						return results, nextCursor, nil
					}
				}
				if entryHasChild(entry) {
					startTable = entryChildID(entry)
				}
				break
			}
			if !entryHasChild(entry) {
				return []PairScanResult{}, nil, nil
			}
			childID := entryChildID(entry)
			if childID == 0 {
				return []PairScanResult{}, nil, nil
			}
			currentTableID = childID
		}
		if startTable == 0 {
			results, nextCursor := acc.finalize(acc.limit)
			return results, nextCursor, nil
		}
	}
	workerCount := 1
	if db.resources != nil {
		workerCount = db.resources.RecommendedWorkers(256)
	}
	if workerCount < 1 {
		workerCount = runtime.NumCPU()
	}
	if workerCount < 1 {
		workerCount = 1
	}
	if err := db.parallelCollectPairEntries(startTable, prefix, workerCount, acc); err != nil {
		return nil, nil, err
	}
	results, nextCursor := acc.finalize(acc.limit)
	return results, nextCursor, nil
}

func (db *Database) PairSummary(prefix []byte, depthLimit int, branchLimit int) (*PairSummaryResult, error) {
	if depthLimit < 0 {
		depthLimit = -1
	}
	if branchLimit < 0 {
		branchLimit = 0
	}
	if branchLimit > pairSummaryMaxBranchLimit {
		branchLimit = pairSummaryMaxBranchLimit
	}
	acc := newPairSummaryAccumulator(prefix, depthLimit, branchLimit)
	startTable := uint32(0)
	if len(prefix) > 0 {
		currentTableID := uint32(0)
		for i, branchByte := range prefix {
			table, err := db.getPairTable(currentTableID)
			if err != nil {
				if os.IsNotExist(err) {
					return acc.finalize(), nil
				}
				return nil, err
			}
			entry, err := table.ReadEntry(branchByte)
			if err != nil {
				return nil, err
			}
			isLast := i == len(prefix)-1
			if isLast {
				if entryHasTerminal(entry) {
					key := decodeAbsoluteKey(entry)
					if key != 0 {
						if err := db.recordSummaryTerminal(acc, append([]byte{}, prefix...), key); err != nil {
							return nil, err
						}
					}
				}
				if entryHasChild(entry) {
					startTable = entryChildID(entry)
				}
				break
			}
			if !entryHasChild(entry) {
				return acc.finalize(), nil
			}
			childID := entryChildID(entry)
			if childID == 0 {
				return acc.finalize(), nil
			}
			currentTableID = childID
		}
		if startTable == 0 {
			return acc.finalize(), nil
		}
	}
	workerCount := 1
	if db.resources != nil {
		workerCount = db.resources.RecommendedWorkers(pairScanDefaultLimit)
	}
	if workerCount < 1 {
		workerCount = runtime.NumCPU()
	}
	if workerCount < 1 {
		workerCount = 1
	}
	if err := db.parallelSummarizePairEntries(startTable, prefix, workerCount, acc); err != nil {
		return nil, err
	}
	return acc.finalize(), nil
}

type pairScanTask struct {
	tableID uint32
	prefix  []byte
}

type pairSummaryTask struct {
	tableID uint32
	path    []byte
}

type pairSummaryAccumulator struct {
	prefix       []byte
	depthLimit   int
	branchLimit  int
	mu           sync.Mutex
	branches     map[string]*pairSummaryBranch
	terminalCnt  int64
	totalBytes   int64
	minPayload   uint32
	maxPayload   uint32
	minKey       uint64
	maxKey       uint64
	maxDepth     int
	selfTerminal bool
}

type pairScanAccumulator struct {
	cursor  []byte
	limit   int
	mu      sync.Mutex
	results []PairScanResult
	count   atomic.Int64
}

func newPairScanAccumulator(limit int, cursor []byte) *pairScanAccumulator {
	return &pairScanAccumulator{
		cursor: append([]byte{}, cursor...),
		limit:  limit,
	}
}

func newPairSummaryAccumulator(prefix []byte, depthLimit int, branchLimit int) *pairSummaryAccumulator {
	return &pairSummaryAccumulator{
		prefix:      append([]byte{}, prefix...),
		depthLimit:  depthLimit,
		branchLimit: branchLimit,
		branches:    make(map[string]*pairSummaryBranch),
	}
}

func (a *pairSummaryAccumulator) recordTerminal(path []byte, key uint64, payloadSize uint32) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.terminalCnt++
	a.totalBytes += int64(payloadSize)
	if a.minPayload == 0 || payloadSize < a.minPayload {
		a.minPayload = payloadSize
	}
	if payloadSize > a.maxPayload {
		a.maxPayload = payloadSize
	}
	if a.minKey == 0 || key < a.minKey {
		a.minKey = key
	}
	if key > a.maxKey {
		a.maxKey = key
	}
	relDepth := len(path) - len(a.prefix)
	if relDepth > a.maxDepth {
		a.maxDepth = relDepth
	}
	if relDepth == 0 {
		a.selfTerminal = true
	}
	if a.depthLimit == 0 || relDepth <= 0 {
		return
	}
	depth := relDepth
	if a.depthLimit > 0 && depth > a.depthLimit {
		depth = a.depthLimit
	}
	if depth <= 0 {
		return
	}
	relPath := path[len(a.prefix) : len(a.prefix)+depth]
	keyStr := hex.EncodeToString(relPath)
	bucket, ok := a.branches[keyStr]
	if !ok {
		bucket = &pairSummaryBranch{Path: append([]byte{}, relPath...)}
		a.branches[keyStr] = bucket
	}
	bucket.Count++
}

func (a *pairSummaryAccumulator) finalize() *PairSummaryResult {
	branches := make([]pairSummaryBranch, 0, len(a.branches))
	for _, branch := range a.branches {
		branches = append(branches, pairSummaryBranch{Path: branch.Path, Count: branch.Count})
	}
	sort.Slice(branches, func(i, j int) bool {
		if branches[i].Count == branches[j].Count {
			return bytes.Compare(branches[i].Path, branches[j].Path) < 0
		}
		return branches[i].Count > branches[j].Count
	})
	if a.branchLimit > 0 && len(branches) > a.branchLimit {
		branches = branches[:a.branchLimit]
	}
	return &PairSummaryResult{
		Prefix:            append([]byte{}, a.prefix...),
		TerminalCount:     a.terminalCnt,
		TotalPayloadBytes: a.totalBytes,
		MinPayloadBytes:   a.minPayload,
		MaxPayloadBytes:   a.maxPayload,
		MinKey:            a.minKey,
		MaxKey:            a.maxKey,
		MaxDepth:          a.maxDepth,
		SelfTerminal:      a.selfTerminal,
		Branches:          branches,
	}
}

func (a *pairScanAccumulator) add(value []byte, key uint64) bool {
	if len(a.cursor) > 0 && bytes.Compare(value, a.cursor) <= 0 {
		return a.shouldStop()
	}
	cp := append([]byte{}, value...)
	a.mu.Lock()
	a.results = append(a.results, PairScanResult{Value: cp, Key: key})
	a.mu.Unlock()
	a.count.Add(1)
	return a.shouldStop()
}

func (a *pairScanAccumulator) shouldStop() bool {
	if a.limit <= 0 {
		return false
	}
	return int(a.count.Load()) >= a.limit
}

func (a *pairScanAccumulator) finalize(limit int) ([]PairScanResult, []byte) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if len(a.results) == 0 {
		return []PairScanResult{}, nil
	}
	sort.Slice(a.results, func(i, j int) bool {
		return bytes.Compare(a.results[i].Value, a.results[j].Value) < 0
	})
	hasMore := limit > 0 && len(a.results) > limit
	if limit > 0 && limit < len(a.results) {
		a.results = a.results[:limit]
	}
	var nextCursor []byte
	if hasMore && len(a.results) > 0 {
		nextCursor = append([]byte{}, a.results[len(a.results)-1].Value...)
	}
	return a.results, nextCursor
}

func (db *Database) parallelCollectPairEntries(tableID uint32, prefix []byte, workers int, acc *pairScanAccumulator) error {
	tasks := make(chan pairScanTask, workers*4)
	var pending sync.WaitGroup
	var workerWG sync.WaitGroup
	var firstErr error
	var errOnce sync.Once
	var abort atomic.Bool

	pending.Add(1)
	tasks <- pairScanTask{tableID: tableID, prefix: append([]byte{}, prefix...)}
	go func() {
		pending.Wait()
		close(tasks)
	}()

	worker := func() {
		defer workerWG.Done()
		for task := range tasks {
			if abort.Load() {
				pending.Done()
				continue
			}
			if err := db.walkPairTable(task, acc, &pending, tasks, &abort); err != nil {
				errOnce.Do(func() {
					firstErr = err
					abort.Store(true)
				})
			}
		}
	}

	for i := 0; i < workers; i++ {
		workerWG.Add(1)
		go worker()
	}
	workerWG.Wait()
	if firstErr != nil {
		return firstErr
	}
	return nil
}

func (db *Database) parallelSummarizePairEntries(tableID uint32, prefix []byte, workers int, acc *pairSummaryAccumulator) error {
	tasks := make(chan pairSummaryTask, workers*4)
	var pending sync.WaitGroup
	var workerWG sync.WaitGroup
	var firstErr error
	var errOnce sync.Once
	var abort atomic.Bool

	pending.Add(1)
	tasks <- pairSummaryTask{tableID: tableID, path: append([]byte{}, prefix...)}
	go func() {
		pending.Wait()
		close(tasks)
	}()

	worker := func() {
		defer workerWG.Done()
		for task := range tasks {
			if abort.Load() {
				pending.Done()
				continue
			}
			if err := db.walkPairSummary(task, acc, &pending, tasks, &abort); err != nil {
				errOnce.Do(func() {
					firstErr = err
					abort.Store(true)
				})
			}
		}
	}

	for i := 0; i < workers; i++ {
		workerWG.Add(1)
		go worker()
	}
	workerWG.Wait()
	if firstErr != nil {
		return firstErr
	}
	return nil
}

func (db *Database) walkPairSummary(
	task pairSummaryTask,
	acc *pairSummaryAccumulator,
	pending *sync.WaitGroup,
	tasks chan<- pairSummaryTask,
	abort *atomic.Bool,
) error {
	defer pending.Done()
	table, err := db.getPairTable(task.tableID)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	for branch := 0; branch < 256; branch++ {
		if abort != nil && abort.Load() {
			return nil
		}
		entry, err := table.ReadEntry(byte(branch))
		if err != nil {
			return err
		}
		if len(entry) == 0 || (!entryHasTerminal(entry) && !entryHasChild(entry)) {
			continue
		}
		value := append(append([]byte{}, task.path...), byte(branch))
		if entryHasTerminal(entry) {
			key := decodeAbsoluteKey(entry)
			if key != 0 {
				if err := db.recordSummaryTerminal(acc, value, key); err != nil {
					return err
				}
			}
		}
		if entryHasChild(entry) {
			childID := entryChildID(entry)
			if childID == 0 {
				continue
			}
			pending.Add(1)
			tasks <- pairSummaryTask{tableID: childID, path: value}
		}
	}
	return nil
}

func (db *Database) recordSummaryTerminal(acc *pairSummaryAccumulator, path []byte, key uint64) error {
	size, err := db.readValueSizeForKey(key)
	if err != nil {
		return err
	}
	acc.recordTerminal(path, key, size)
	return nil
}

func (db *Database) walkPairTable(
	task pairScanTask,
	acc *pairScanAccumulator,
	pending *sync.WaitGroup,
	tasks chan<- pairScanTask,
	abort *atomic.Bool,
) error {
	defer pending.Done()
	table, err := db.getPairTable(task.tableID)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	for branch := 0; branch < 256; branch++ {
		if abort != nil && abort.Load() {
			return nil
		}
		entry, err := table.ReadEntry(byte(branch))
		if err != nil {
			return err
		}
		value := append(append([]byte{}, task.prefix...), byte(branch))
		if entryHasTerminal(entry) {
			if acc.add(value, decodeAbsoluteKey(entry)) && abort != nil && acc.limit > 0 {
				abort.Store(true)
				return nil
			}
		}
		if entryHasChild(entry) {
			childID := entryChildID(entry)
			if childID == 0 {
				continue
			}
			pending.Add(1)
			tasks <- pairScanTask{tableID: childID, prefix: append([]byte{}, value...)}
		}
	}
	return nil
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

func (db *Database) readValueSizeForKey(key uint64) (uint32, error) {
	entry, err := db.mainKeys.ReadEntry(key)
	if err != nil {
		return 0, err
	}
	size := readValueSize(entry)
	if size == 0 {
		return 0, fmt.Errorf("key %d has no payload", key)
	}
	return size, nil
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

func formatPairSummaryResponse(res *PairSummaryResult) string {
	if res == nil {
		return "ERROR,summary_unavailable"
	}
	var b strings.Builder
	b.WriteString("SUCCESS,command=PAIR_SUMMARY")
	b.WriteString(fmt.Sprintf(",count=%d", res.TerminalCount))
	b.WriteString(fmt.Sprintf(",total_payload_bytes=%d", res.TotalPayloadBytes))
	b.WriteString(fmt.Sprintf(",min_payload_bytes=%d", res.MinPayloadBytes))
	b.WriteString(fmt.Sprintf(",max_payload_bytes=%d", res.MaxPayloadBytes))
	if res.MinKey > 0 {
		b.WriteString(fmt.Sprintf(",min_key=%d", res.MinKey))
	}
	if res.MaxKey > 0 {
		b.WriteString(fmt.Sprintf(",max_key=%d", res.MaxKey))
	}
	b.WriteString(fmt.Sprintf(",max_depth=%d", res.MaxDepth))
	b.WriteString(fmt.Sprintf(",self_terminal=%d", boolToInt(res.SelfTerminal)))
	branchCount := len(res.Branches)
	if branchCount > 0 {
		parts := make([]string, 0, branchCount)
		for i := 0; i < branchCount; i++ {
			parts = append(parts, fmt.Sprintf("%x:%d", res.Branches[i].Path, res.Branches[i].Count))
		}
		b.WriteString(fmt.Sprintf(",branch_count=%d", branchCount))
		b.WriteString(fmt.Sprintf(",branches=%s", strings.Join(parts, ";")))
	} else {
		b.WriteString(",branch_count=0")
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

func parseFileCheckpointArgs(raw string) (FileCheckpointOptions, error) {
	opts := FileCheckpointOptions{}
	trimmed := strings.TrimSpace(raw)
	if trimmed == "" {
		return opts, nil
	}
	tokens := strings.Fields(trimmed)
	for _, token := range tokens {
		upper := strings.ToUpper(token)
		switch {
		case upper == "DROP_CACHE":
			opts.DisableCache = true
		case upper == "CLOSE_HANDLES":
			opts.CloseHandles = true
		case strings.HasPrefix(upper, "IDLE="):
			value := strings.TrimSpace(token[5:])
			if value == "" {
				return opts, fmt.Errorf("invalid_checkpoint_option:%s", token)
			}
			duration, err := time.ParseDuration(value)
			if err != nil {
				return opts, fmt.Errorf("invalid_idle_duration:%s", value)
			}
			if duration < 0 {
				duration = 0
			}
			opts.IdleThreshold = duration
		default:
			if duration, err := time.ParseDuration(token); err == nil {
				if duration < 0 {
					duration = 0
				}
				opts.IdleThreshold = duration
				continue
			}
			if seconds, err := strconv.Atoi(token); err == nil {
				if seconds < 0 {
					seconds = 0
				}
				opts.IdleThreshold = time.Duration(seconds) * time.Second
				continue
			}
			return opts, fmt.Errorf("invalid_checkpoint_option:%s", token)
		}
	}
	return opts, nil
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
	var cacheStats *payloadCacheStats
	if db.payloadCache != nil {
		stats := db.payloadCache.Stats()
		cacheStats = &stats
	}
	return formatSystemStatsResponse(db.resources.Snapshot(), cacheStats)
}

func formatSystemStatsResponse(snap ResourceSnapshot, cache *payloadCacheStats) string {
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
	if len(snap.WorkerHints) > 0 {
		keys := make([]int, 0, len(snap.WorkerHints))
		for pending := range snap.WorkerHints {
			keys = append(keys, pending)
		}
		sort.Ints(keys)
		parts := make([]string, 0, len(keys))
		for _, pending := range keys {
			parts = append(parts, fmt.Sprintf("%d:%d", pending, snap.WorkerHints[pending]))
		}
		b.WriteString(fmt.Sprintf(",recommended_workers=%s", strings.Join(parts, ";")))
	}
	if cache != nil {
		b.WriteString(",payload_cache_enabled=1")
		b.WriteString(fmt.Sprintf(",payload_cache_entries=%d", cache.Entries))
		b.WriteString(fmt.Sprintf(",payload_cache_max_entries=%d", cache.MaxEntries))
		b.WriteString(fmt.Sprintf(",payload_cache_bytes=%d", cache.Bytes))
		b.WriteString(fmt.Sprintf(",payload_cache_max_bytes=%d", cache.MaxBytes))
		b.WriteString(fmt.Sprintf(",payload_cache_hits=%d", cache.Hits))
		b.WriteString(fmt.Sprintf(",payload_cache_misses=%d", cache.Misses))
		b.WriteString(fmt.Sprintf(",payload_cache_evictions=%d", cache.Evictions))
		if cache.CalculatedHitRatioPct > 0 {
			b.WriteString(fmt.Sprintf(",payload_cache_hit_pct=%.2f", cache.CalculatedHitRatioPct))
		} else {
			b.WriteString(",payload_cache_hit_pct=0")
		}
		if cache.AdvisoryBypassBytes > 0 {
			b.WriteString(fmt.Sprintf(",payload_cache_advisory_bypass_bytes=%d", cache.AdvisoryBypassBytes))
		}
	} else {
		b.WriteString(",payload_cache_enabled=0")
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
