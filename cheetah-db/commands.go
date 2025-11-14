// commands.go
package main

import (
	"fmt"
	"io"
	"os"
	"sort"
	"sync"
)

const maxValueSize = int(^uint32(0))

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
	for i, branchByte := range value {
		table, err := db.getPairTable(currentTableID)
		if err != nil {
			return "", err
		}
		entry, _ := table.ReadEntry(branchByte)
		isLast := i == len(value)-1
		if isLast {
			setEntryTerminal(entry, absKey)
			if err := table.WriteEntry(branchByte, entry); err != nil {
				return "", err
			}
			return "SUCCESS,pair_set", nil
		}
		childID := entryChildID(entry)
		if !entryHasChild(entry) || childID == 0 {
			newID, err := db.getNewPairTableID()
			if err != nil {
				return "", err
			}
			setEntryChild(entry, newID)
			if err := table.WriteEntry(branchByte, entry); err != nil {
				return "", err
			}
			currentTableID = newID
			continue
		}
		currentTableID = childID
	}
	return "ERROR,internal_logic_error", nil
}

func (db *Database) PairGet(value []byte) (string, error) {
	if len(value) == 0 {
		return "ERROR,pair_value_cannot_be_empty", nil
	}

	currentTableID := uint32(0)
	for i, branchByte := range value {
		table, err := db.getPairTable(currentTableID)
		if err != nil {
			if os.IsNotExist(err) {
				return "ERROR,not_found", nil
			}
			return "", err
		}

		entry, _ := table.ReadEntry(branchByte)
		isLast := i == len(value)-1
		if isLast {
			if !entryHasTerminal(entry) {
				return "ERROR,not_found", nil
			}
			absKey := decodeAbsoluteKey(entry)
			return fmt.Sprintf("SUCCESS,key=%d", absKey), nil
		}
		if !entryHasChild(entry) {
			return "ERROR,not_found", nil
		}
		currentTableID = entryChildID(entry)
		if currentTableID == 0 {
			return "ERROR,not_found", nil
		}
	}
	return "ERROR,not_found", nil
}

// PairDel cancella una mappatura valore->chiave e pulisce i nodi orfani.

type pathStackFrame struct {
	TableID    uint32
	BranchByte byte
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
	for i, branchByte := range value {
		table, err := db.getPairTable(currentTableID)
		if err != nil {
			if os.IsNotExist(err) {
				return "ERROR,not_found", nil
			}
			return "", err
		}

		table.mu.RLock()
		acquiredLocks = append(acquiredLocks, &table.mu)

		entry, err := table.ReadEntry(branchByte)
		if err != nil {
			cleanupReadLocks()
			return "ERROR,not_found", nil
		}

		pathStack = append(pathStack, pathStackFrame{TableID: currentTableID, BranchByte: branchByte})
		isLast := i == len(value)-1
		if isLast {
			if !entryHasTerminal(entry) {
				cleanupReadLocks()
				return "ERROR,not_found", nil
			}
		} else {
			if !entryHasChild(entry) {
				cleanupReadLocks()
				return "ERROR,not_found", nil
			}
			currentTableID = entryChildID(entry)
		}
	}
	cleanupReadLocks() // Rilasciamo tutti i read lock prima di passare ai write lock

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
	terminalEntry, _ := terminalTable.ReadEntry(terminalFrame.BranchByte)
	if !entryHasTerminal(terminalEntry) {
		return "ERROR,not_found", nil
	}
	clearEntryTerminal(terminalEntry)
	if err := terminalTable.WriteEntry(terminalFrame.BranchByte, terminalEntry); err != nil {
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
		parentEntry, _ := parentTable.ReadEntry(parentFrame.BranchByte)
		clearEntryChild(parentEntry)
		if err := parentTable.WriteEntry(parentFrame.BranchByte, parentEntry); err != nil {
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
