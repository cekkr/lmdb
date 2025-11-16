// commands.go
package main

import (
	"errors"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
)

const maxValueSize = int(^uint32(0))

var errPairTraversalAbort = errors.New("pair traversal aborted")

// --- Metodi CRUD per Database ---

func (db *Database) Insert(value []byte, specifiedSize int) (string, error) {
	valueSize := len(value)
	if specifiedSize > 0 && valueSize != specifiedSize {
		return fmt.Sprintf("ERROR,value_size_mismatch (expected %d, got %d)", specifiedSize, valueSize), nil
	}
	if valueSize <= 0 || valueSize > maxValueSize {
		return "ERROR,invalid_value_size", nil
	}

	sizeField := uint32(valueSize)
	location, err := db.getAvailableLocation(sizeField)
	if err != nil {
		return "ERROR,cannot_get_value_location", err
	}

	vTable, err := db.getValuesTable(sizeField, location.TableID)
	if err != nil {
		return "ERROR,cannot_load_values_table", err
	}
	offset := int64(location.EntryID) * int64(sizeField)
	if _, err := vTable.WriteAt(value, offset); err != nil {
		return "ERROR,value_write_failed", err
	}
	db.cachePayload(sizeField, location, value)

	newKey := db.highestKey.Add(1)
	entry := make([]byte, MainKeysEntrySize)
	writeValueSize(entry, sizeField)
	copy(entry[ValueSizeBytes:], location.Encode())

	if err := db.mainKeys.WriteEntry(newKey, entry); err != nil {
		db.highestKey.Add(^uint64(0)) // Rollback del contatore in caso di errore
		return "ERROR,key_write_failed", err
	}

	return fmt.Sprintf("SUCCESS,key=%d", newKey), nil
}

func (db *Database) Read(key uint64) (string, error) {
	entry, err := db.mainKeys.ReadEntry(key)
	if err != nil {
		if os.IsNotExist(err) || err == io.EOF {
			return "ERROR,key_not_found", nil
		}
		return "ERROR,key_read_failed", err
	}

	valueSize := readValueSize(entry)
	if valueSize == 0 {
		return "ERROR,key_not_found (deleted)", nil
	}
	location := DecodeValueLocationIndex(entry[ValueSizeBytes:])

	if cached, ok := db.getCachedPayload(valueSize, location); ok {
		return fmt.Sprintf("SUCCESS,size=%d,value=%s", valueSize, string(cached)), nil
	}

	vTable, err := db.getValuesTable(valueSize, location.TableID)
	if err != nil {
		return "ERROR,cannot_load_values_table", err
	}
	value := make([]byte, int(valueSize))
	offset := int64(location.EntryID) * int64(valueSize)
	if _, err := vTable.ReadAt(value, offset); err != nil {
		return "ERROR,value_read_failed", err
	}
	db.cachePayload(valueSize, location, value)

	return fmt.Sprintf("SUCCESS,size=%d,value=%s", valueSize, string(value)), nil
}

func (db *Database) Edit(key uint64, newValue []byte) (string, error) {
	entry, err := db.mainKeys.ReadEntry(key)
	if err != nil {
		return "ERROR,key_not_found", err
	}
	valueSize := readValueSize(entry)
	if valueSize == 0 {
		return "ERROR,key_not_found (deleted)", nil
	}
	if len(newValue) != int(valueSize) {
		return fmt.Sprintf("ERROR,value_size_mismatch (expected %d, got %d)", valueSize, len(newValue)), nil
	}

	location := DecodeValueLocationIndex(entry[ValueSizeBytes:])
	vTable, err := db.getValuesTable(valueSize, location.TableID)
	if err != nil {
		return "ERROR,cannot_load_values_table", err
	}
	offset := int64(location.EntryID) * int64(valueSize)
	if _, err := vTable.WriteAt(newValue, offset); err != nil {
		return "ERROR,value_update_failed", err
	}
	db.cachePayload(valueSize, location, newValue)

	return fmt.Sprintf("SUCCESS,key=%d_updated", key), nil
}

func (db *Database) Delete(key uint64) (string, error) {
	lock := db.mainKeys.getLock(key)
	lock.Lock()
	defer lock.Unlock()

	entry, err := db.mainKeys.readEntryFromFile(key) // Usa il metodo interno non bloccante
	if err != nil {
		return "ERROR,key_not_found", err
	}
	valueSize := readValueSize(entry)
	if valueSize == 0 {
		return "ERROR,already_deleted", nil
	}

	// Aggiungi l'indice alla tabella di riciclo
	locationBytes := make([]byte, ValueLocationIndexSize)
	copy(locationBytes, entry[ValueSizeBytes:])
	location := DecodeValueLocationIndex(locationBytes)
	db.invalidatePayload(valueSize, location)

	rTable, err := db.getRecycleTable(valueSize)
	if err != nil {
		return "ERROR,cannot_load_recycle_table", err
	}
	if err := rTable.Push(locationBytes); err != nil {
		return "ERROR,recycle_failed", err
	}

	// Azzera la chiave nella tabella principale
	if err := db.mainKeys.writeEntryToFile(key, make([]byte, MainKeysEntrySize)); err != nil {
		// Qui servirebbe un rollback del Push, ma per ora lo omettiamo
		return "ERROR,key_delete_failed", err
	}

	// Se abbiamo eliminato la chiave più alta, trova la nuova
	if key == db.highestKey.Load() {
		db.findNewHighestKey(key)
	}

	return fmt.Sprintf("SUCCESS,key=%d_deleted", key), nil
}

func (db *Database) PairSet(value []byte, absKey uint64) (string, error) {
	if len(value) == 0 {
		return "ERROR,pair_value_cannot_be_empty", nil
	}

	currentTableID := uint32(0)
	err := db.branchCodec.walkKey(value, func(index uint32, chunk []byte, isLast bool) error {
		table, err := db.getPairTable(currentTableID)
		if err != nil {
			return err
		}
		entry, _ := table.ReadEntry(index)
		if isLast {
			setEntryTerminal(entry, absKey)
			return table.WriteEntry(index, entry)
		}
		childID := entryChildID(entry)
		if !entryHasChild(entry) || childID == 0 {
			newID, err := db.getNewPairTableID()
			if err != nil {
				return err
			}
			setEntryChild(entry, newID)
			if err := table.WriteEntry(index, entry); err != nil {
				return err
			}
			currentTableID = newID
			return nil
		}
		currentTableID = childID
		return nil
	})
	if err != nil {
		return "", err
	}
	return "SUCCESS,pair_set", nil
}

func (db *Database) PairGet(value []byte) (string, error) {
	if len(value) == 0 {
		return "ERROR,pair_value_cannot_be_empty", nil
	}

	currentTableID := uint32(0)
	var result string
	err := db.branchCodec.walkKey(value, func(index uint32, chunk []byte, isLast bool) error {
		table, err := db.getPairTable(currentTableID)
		if err != nil {
			if os.IsNotExist(err) {
				result = "ERROR,not_found"
				return errPairTraversalAbort
			}
			return err
		}
		entry, _ := table.ReadEntry(index)
		if isLast {
			if !entryHasTerminal(entry) {
				result = "ERROR,not_found"
				return errPairTraversalAbort
			}
			result = fmt.Sprintf("SUCCESS,key=%d", decodeAbsoluteKey(entry))
			return errPairTraversalAbort
		}
		if !entryHasChild(entry) {
			result = "ERROR,not_found"
			return errPairTraversalAbort
		}
		currentTableID = entryChildID(entry)
		if currentTableID == 0 {
			result = "ERROR,not_found"
			return errPairTraversalAbort
		}
		return nil
	})
	if err != nil && err != errPairTraversalAbort {
		return "", err
	}
	if result == "" {
		result = "ERROR,not_found"
	}
	return result, nil
}

// PairDel cancella una mappatura valore->chiave e pulisce i nodi orfani.

type pathStackFrame struct {
	TableID     uint32
	BranchIndex uint32
}

// PairDel cancella una mappatura valore->chiave e pulisce i nodi orfani,
// implementando la compressione del percorso.
func (db *Database) PairDel(value []byte) (string, error) {
	if len(value) == 0 {
		return "ERROR,pair_value_cannot_be_empty", nil
	}

	// --- Fase 1: Scoperta del Percorso (Read Locks) ---
	pathStack := make([]pathStackFrame, 0, len(value))
	acquiredLocks := make([]*sync.RWMutex, 0, len(value))

	cleanupReadLocks := func() {
		for i := len(acquiredLocks) - 1; i >= 0; i-- {
			acquiredLocks[i].RUnlock()
		}
	}

	currentTableID := uint32(0)
	err := db.branchCodec.walkKey(value, func(index uint32, chunk []byte, isLast bool) error {
		table, err := db.getPairTable(currentTableID)
		if err != nil {
			if os.IsNotExist(err) {
				return errPairTraversalAbort
			}
			return err
		}

		table.mu.RLock()
		acquiredLocks = append(acquiredLocks, &table.mu)

		entry, err := table.ReadEntry(index)
		if err != nil {
			return errPairTraversalAbort
		}

		pathStack = append(pathStack, pathStackFrame{TableID: currentTableID, BranchIndex: index})
		if isLast {
			if !entryHasTerminal(entry) {
				return errPairTraversalAbort
			}
		} else {
			if !entryHasChild(entry) {
				return errPairTraversalAbort
			}
			currentTableID = entryChildID(entry)
		}
		return nil
	})
	cleanupReadLocks()
	if err != nil {
		if err == errPairTraversalAbort {
			return "ERROR,not_found", nil
		}
		return "", err
	}

	// --- Fase 2: Blocco Esclusivo del Percorso (Write Locks) ---
	pathTableIDs := make([]uint32, len(pathStack))
	for i, frame := range pathStack {
		pathTableIDs[i] = frame.TableID
	}

	// Ordiniamo gli ID per acquisire i lock in ordine ed evitare deadlock
	sort.Slice(pathTableIDs, func(i, j int) bool { return pathTableIDs[i] < pathTableIDs[j] })

	lockedTables := make(map[uint32]*PairTable)
	for _, id := range pathTableIDs {
		table, _ := db.getPairTable(id)
		table.mu.Lock()
		lockedTables[id] = table
	}
	defer func() { // Assicuriamo il rilascio di tutti i write lock
		for _, id := range pathTableIDs {
			if table, ok := lockedTables[id]; ok {
				table.mu.Unlock()
			}
		}
	}()

	// --- Fase 3: Modifica e Pulizia ---
	// Modifichiamo il nodo terminale
	terminalFrame := pathStack[len(pathStack)-1]
	terminalTable := lockedTables[terminalFrame.TableID]
	terminalEntry, _ := terminalTable.ReadEntry(terminalFrame.BranchIndex)
	if !entryHasTerminal(terminalEntry) {
		return "ERROR,not_found", nil
	}
	clearEntryTerminal(terminalEntry)
	if err := terminalTable.WriteEntry(terminalFrame.BranchIndex, terminalEntry); err != nil {
		return "", fmt.Errorf("failed to clear terminal entry: %w", err)
	}

	// Risaliamo lo stack per pulire eventuali nodi vuoti
	for i := len(pathStack) - 1; i >= 0; i-- {
		frame := pathStack[i]
		tableToAnalyze := lockedTables[frame.TableID]
		isEmpty, err := tableToAnalyze.IsEmpty()
		if err != nil || !isEmpty {
			break
		}
		if i == 0 {
			break // non cancelliamo mai la radice
		}
		parentFrame := pathStack[i-1]
		parentTable := lockedTables[parentFrame.TableID]
		parentEntry, _ := parentTable.ReadEntry(parentFrame.BranchIndex)
		clearEntryChild(parentEntry)
		if err := parentTable.WriteEntry(parentFrame.BranchIndex, parentEntry); err != nil {
			return "", err
		}
		if frame.TableID != 0 {
			if err := db.deletePairTable(frame.TableID); err != nil {
				return "", err
			}
			delete(lockedTables, frame.TableID)
		}
	}

	// --- Fase 4: Rilascio dei Lock ---
	// Gestito dal defer in cima alla fase 2

	return "SUCCESS,pair_deleted", nil
}

// PairPurge removes every pair entry beneath the provided prefix and deletes the
// associated payload keys in bulk. It returns the number of entries cleared.
func (db *Database) PairPurge(prefix []byte, limit int) (int, error) {
	if limit <= 0 {
		limit = pairScanMaxLimit
	} else {
		limit = normalizePairScanLimit(limit)
	}
	var cursor []byte
	totalRemoved := 0
	for {
		results, nextCursor, err := db.PairScan(prefix, limit, cursor)
		if err != nil {
			return totalRemoved, err
		}
		if len(results) == 0 {
			break
		}
		removed, err := db.purgePairEntries(results)
		totalRemoved += removed
		if err != nil {
			return totalRemoved, err
		}
		cursor = nextCursor
		if cursor == nil && len(results) < limit {
			break
		}
	}
	return totalRemoved, nil
}

func (db *Database) purgePairEntries(results []PairScanResult) (int, error) {
	if len(results) == 0 {
		return 0, nil
	}
	workerCount := len(results)
	if db.resources != nil {
		if recommended := db.resources.RecommendedWorkers(len(results)); recommended > 0 {
			workerCount = recommended
		}
	}
	if workerCount < 1 {
		workerCount = 1
	}

	sem := make(chan struct{}, workerCount)
	var wg sync.WaitGroup
	var removed atomic.Int64
	var firstErr error
	var errOnce sync.Once

	for _, res := range results {
		value := append([]byte{}, res.Value...)
		key := res.Key
		wg.Add(1)
		go func(val []byte, absKey uint64) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()
			if err := db.purgePairEntry(val, absKey); err != nil {
				errOnce.Do(func() { firstErr = err })
				return
			}
			removed.Add(1)
		}(value, key)
	}

	wg.Wait()
	if firstErr != nil {
		return int(removed.Load()), firstErr
	}
	return int(removed.Load()), nil
}

func (db *Database) purgePairEntry(value []byte, key uint64) error {
	resp, err := db.Delete(key)
	if err != nil {
		if !isDeleteResponseIgnorable(resp) {
			return fmt.Errorf("delete key %d failed: %w", key, err)
		}
	} else if !isDeleteResponseIgnorable(resp) {
		return fmt.Errorf("delete key %d failed: %s", key, resp)
	}

	resp, err = db.PairDel(value)
	if err != nil {
		return fmt.Errorf("pair delete %x failed: %w", value, err)
	}
	if !isPairDelResponseIgnorable(resp) {
		return fmt.Errorf("pair delete %x failed: %s", value, resp)
	}
	return nil
}

func isDeleteResponseIgnorable(resp string) bool {
	if resp == "" || strings.HasPrefix(resp, "SUCCESS") {
		return true
	}
	lower := strings.ToLower(resp)
	return strings.Contains(lower, "already_deleted") || strings.Contains(lower, "key_not_found")
}

func isPairDelResponseIgnorable(resp string) bool {
	if resp == "" || strings.HasPrefix(resp, "SUCCESS") {
		return true
	}
	return strings.Contains(strings.ToLower(resp), "not_found")
}
