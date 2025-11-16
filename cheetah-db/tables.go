// tables.go
package main

import (
	"encoding/binary"
	"hash/fnv"
	"io"
	"os"
	"sync"
	"sync/atomic"
	"time"
)

//const KeyStripeCount = 1024 // in types.go

// --- MainKeysTable ---

// MainKeysTable gestisce l'accesso al file main_keys.table
type MainKeysTable struct {
	file  *os.File
	path  string
	locks []sync.RWMutex // Lock striping
}

func NewMainKeysTable(path string) (*MainKeysTable, error) {
	file, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE, 0644)
	if err != nil {
		return nil, err
	}
	return &MainKeysTable{
		file:  file,
		path:  path,
		locks: make([]sync.RWMutex, KeyStripeCount),
	}, nil
}

func (t *MainKeysTable) getLock(key uint64) *sync.RWMutex {
	hasher := fnv.New64a()
	hasher.Write(binary.BigEndian.AppendUint64(nil, key))
	return &t.locks[hasher.Sum64()%KeyStripeCount]
}

func (t *MainKeysTable) ReadEntry(key uint64) ([]byte, error) {
	lock := t.getLock(key)
	lock.RLock()
	defer lock.RUnlock()

	entry := make([]byte, MainKeysEntrySize)
	_, err := t.file.ReadAt(entry, int64(key)*MainKeysEntrySize)
	return entry, err
}

func (t *MainKeysTable) WriteEntry(key uint64, entry []byte) error {
	lock := t.getLock(key)
	lock.Lock()
	defer lock.Unlock()

	_, err := t.file.WriteAt(entry, int64(key)*MainKeysEntrySize)
	return err
}

func (t *MainKeysTable) Close() {
	t.file.Close()
}

// Metodi senza lock (per uso interno quando il lock è già acquisito)
func (t *MainKeysTable) readEntryFromFile(key uint64) ([]byte, error) {
	entry := make([]byte, MainKeysEntrySize)
	_, err := t.file.ReadAt(entry, int64(key)*MainKeysEntrySize)
	return entry, err
}
func (t *MainKeysTable) writeEntryToFile(key uint64, entry []byte) error {
	_, err := t.file.WriteAt(entry, int64(key)*MainKeysEntrySize)
	return err
}

// --- ValuesTable ---
type ValuesTable struct {
	file *os.File
	mu   sync.RWMutex
}

func NewValuesTable(path string) (*ValuesTable, error) {
	file, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE, 0644)
	if err != nil {
		return nil, err
	}
	return &ValuesTable{file: file}, nil
}

func (t *ValuesTable) WriteAt(p []byte, off int64) (n int, err error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.file.WriteAt(p, off)
}

func (t *ValuesTable) ReadAt(p []byte, off int64) (n int, err error) {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.file.ReadAt(p, off)
}

func (t *ValuesTable) Close() {
	t.file.Close()
}

// --- RecycleTable ---
type RecycleTable struct {
	file *os.File
	mu   sync.Mutex
}

func NewRecycleTable(path string) (*RecycleTable, error) {
	file, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE, 0644)
	if err != nil {
		return nil, err
	}
	return &RecycleTable{file: file}, nil
}

func (t *RecycleTable) Close() { t.file.Close() }

func (t *RecycleTable) Pop() ([]byte, bool) {
	t.mu.Lock()
	defer t.mu.Unlock()

	counterBytes := make([]byte, RecycleCounterSize)
	if _, err := t.file.ReadAt(counterBytes, 0); err != nil {
		return nil, false // File vuoto o errore
	}
	count := binary.BigEndian.Uint16(counterBytes)
	if count == 0 {
		return nil, false
	}

	offset := int64(RecycleCounterSize) + int64(count-1)*ValueLocationIndexSize
	locBytes := make([]byte, ValueLocationIndexSize)
	if _, err := t.file.ReadAt(locBytes, offset); err != nil {
		return nil, false
	}

	binary.BigEndian.PutUint16(counterBytes, count-1)
	if _, err := t.file.WriteAt(counterBytes, 0); err != nil {
		// Errore critico, ma l'indice è stato letto. Potremmo loggarlo.
	}
	return locBytes, true
}

func (t *RecycleTable) Push(locationBytes []byte) error {
	t.mu.Lock()
	defer t.mu.Unlock()

	counterBytes := make([]byte, RecycleCounterSize)
	count := uint16(0)
	if _, err := t.file.ReadAt(counterBytes, 0); err == nil {
		count = binary.BigEndian.Uint16(counterBytes)
	}

	offset := int64(RecycleCounterSize) + int64(count)*ValueLocationIndexSize
	if _, err := t.file.WriteAt(locationBytes, offset); err != nil {
		return err
	}

	binary.BigEndian.PutUint16(counterBytes, count+1)
	_, err := t.file.WriteAt(counterBytes, 0)
	return err
}

// /
// / --- PairTable (TreeTable Node) ---
// /
type PairTable struct {
	id       uint32
	path     string
	file     *os.File
	mu       sync.RWMutex
	handleMu sync.Mutex
	refCount atomic.Int32
	lastUsed atomic.Int64
	db       *Database
}

func NewPairTable(db *Database, tableID uint32, path string) (*PairTable, error) {
	file, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE, 0644)
	if err != nil {
		return nil, err
	}

	info, err := file.Stat()
	if err != nil {
		file.Close()
		return nil, err
	}
	if info.Size() == 0 {
		if err := file.Truncate(int64(PairTablePreallocatedSize)); err != nil {
			file.Close()
			return nil, err
		}
	}

	pt := &PairTable{id: tableID, path: path, file: file, db: db}
	pt.touch()
	pt.notifyHandleOpened()
	return pt, nil
}

func (t *PairTable) notifyHandleOpened() {
	if t.db != nil {
		t.db.onPairTableHandleOpened()
	}
}

func (t *PairTable) ensureHandle() error {
	t.handleMu.Lock()
	defer t.handleMu.Unlock()
	if t.file != nil {
		return nil
	}
	file, err := os.OpenFile(t.path, os.O_RDWR|os.O_CREATE, 0644)
	if err != nil {
		return err
	}
	t.file = file
	t.notifyHandleOpened()
	return nil
}

func (t *PairTable) closeHandleIfIdle() bool {
	t.handleMu.Lock()
	defer t.handleMu.Unlock()
	if t.file == nil || t.refCount.Load() > 0 {
		return false
	}
	t.file.Close()
	t.file = nil
	if t.db != nil {
		t.db.onPairTableHandleClosed()
	}
	return true
}

func (t *PairTable) forceCloseHandle() {
	t.handleMu.Lock()
	defer t.handleMu.Unlock()
	if t.file == nil {
		return
	}
	t.file.Close()
	t.file = nil
	if t.db != nil {
		t.db.onPairTableHandleClosed()
	}
}

func (t *PairTable) hasHandle() bool {
	t.handleMu.Lock()
	defer t.handleMu.Unlock()
	return t.file != nil
}

func (t *PairTable) beginUse() {
	t.refCount.Add(1)
	t.touch()
}

func (t *PairTable) endUse() {
	t.refCount.Add(-1)
}

func (t *PairTable) touch() {
	t.lastUsed.Store(time.Now().UnixNano())
}

func (t *PairTable) InUse() bool {
	return t.refCount.Load() > 0
}

func (t *PairTable) LastUsedUnixNano() int64 {
	return t.lastUsed.Load()
}

func (t *PairTable) ReadEntry(branchByte byte) ([]byte, error) {
	t.beginUse()
	defer t.endUse()
	if err := t.ensureHandle(); err != nil {
		return nil, err
	}

	t.mu.RLock()
	defer t.mu.RUnlock()

	entry := make([]byte, PairEntrySize)
	offset := int64(branchByte) * int64(PairEntrySize)
	_, err := t.file.ReadAt(entry, offset)
	return entry, err
}

func (t *PairTable) WriteEntry(branchByte byte, entry []byte) error {
	t.beginUse()
	defer t.endUse()
	if err := t.ensureHandle(); err != nil {
		return err
	}

	t.mu.Lock()
	defer t.mu.Unlock()

	offset := int64(branchByte) * int64(PairEntrySize)
	_, err := t.file.WriteAt(entry, offset)
	return err
}

// IsEmpty controlla se il nodo non ha più figli o chiavi terminali.
func (t *PairTable) IsEmpty() (bool, error) {
	t.beginUse()
	defer t.endUse()
	if err := t.ensureHandle(); err != nil {
		return false, err
	}

	t.mu.RLock()
	defer t.mu.RUnlock()

	info, err := t.file.Stat()
	if err != nil {
		if os.IsNotExist(err) {
			return true, nil
		}
		return false, err
	}
	if info.Size() == 0 {
		return true, nil
	}

	buffer := make([]byte, info.Size())
	if _, err := t.file.ReadAt(buffer, 0); err != nil && err != io.EOF {
		return false, err
	}

	for i := 0; i < len(buffer); i += PairEntrySize {
		if buffer[i] != 0 {
			return false, nil
		}
	}
	return true, nil
}

func (t *PairTable) Close() {
	deadline := time.Now().Add(250 * time.Millisecond)
	for t.InUse() && time.Now().Before(deadline) {
		time.Sleep(5 * time.Millisecond)
	}
	t.forceCloseHandle()
}

// Snapshot returns a full copy of the current table state so callers can scan without holding locks.
func (t *PairTable) Snapshot() ([]byte, error) {
	t.beginUse()
	defer t.endUse()
	if err := t.ensureHandle(); err != nil {
		return nil, err
	}

	t.mu.RLock()
	defer t.mu.RUnlock()

	buf := make([]byte, PairTablePreallocatedSize)
	if _, err := t.file.ReadAt(buf, 0); err != nil && err != io.EOF {
		return nil, err
	}
	return buf, nil
}

// Path restituisce il percorso del file della tabella.
func (t *PairTable) Path() string {
	return t.path
}
