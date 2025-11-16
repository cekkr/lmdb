package main

import (
	"errors"
	"io"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

const (
	defaultSectorSize       = 4096
	defaultMaxCachedSectors = 64
	flushQueueSize          = 1024
)

type ManagedFileOptions struct {
	PreallocateSize  int64
	CacheEnabled     bool
	FlushInterval    time.Duration
	SectorSize       int64
	MaxCachedSectors int
}

type flushRequest struct {
	file *ManagedFile
	key  uint64
}

type FileCheckpointOptions struct {
	IdleThreshold time.Duration
	DisableCache  bool
	CloseHandles  bool
}

type FileManager struct {
	limit        int
	files        sync.Map
	openHandles  atomic.Int32
	flushQueue   chan flushRequest
	flushStop    chan struct{}
	flushWG      sync.WaitGroup
	flushWorkers int
}

func NewFileManager(limit int) *FileManager {
	if limit < 0 {
		limit = 0
	}
	manager := &FileManager{limit: limit}
	manager.startFlusher()
	return manager
}

func (fm *FileManager) register(file *ManagedFile) {
	fm.files.Store(file.path, file)
}

func (fm *FileManager) handleOpened() {
	fm.openHandles.Add(1)
	fm.enforceLimit()
}

func (fm *FileManager) handleClosed() {
	if fm.openHandles.Load() > 0 {
		fm.openHandles.Add(-1)
	}
}

func (fm *FileManager) startFlusher() {
	workers := fm.resolveFlushWorkers()
	fm.flushWorkers = workers
	if workers <= 0 {
		return
	}
	fm.flushQueue = make(chan flushRequest, flushQueueSize)
	fm.flushStop = make(chan struct{})
	for i := 0; i < workers; i++ {
		fm.flushWG.Add(1)
		go fm.flushWorker()
	}
}

func (fm *FileManager) resolveFlushWorkers() int {
	if raw := strings.TrimSpace(os.Getenv("CHEETAH_FLUSH_WORKERS")); raw != "" {
		if val, err := strconv.Atoi(raw); err == nil && val > 0 {
			return val
		}
	}
	workers := runtime.NumCPU()
	if workers < 1 {
		workers = 1
	}
	return workers
}

func (fm *FileManager) flushWorker() {
	defer fm.flushWG.Done()
	for {
		select {
		case req := <-fm.flushQueue:
			if req.file != nil {
				req.file.flushSector(req.key)
			}
		case <-fm.flushStop:
			return
		}
	}
}

func (fm *FileManager) Close() {
	if fm == nil {
		return
	}
	fm.ForceCheckpoint(FileCheckpointOptions{
		DisableCache: true,
		CloseHandles: true,
	})
	if fm.flushStop != nil {
		close(fm.flushStop)
	}
	fm.flushWG.Wait()
}

func (fm *FileManager) enqueueFlush(file *ManagedFile, key uint64) bool {
	if fm.flushQueue == nil {
		return false
	}
	select {
	case fm.flushQueue <- flushRequest{file: file, key: key}:
		return true
	default:
		go file.flushSector(key)
		return true
	}
}

func (fm *FileManager) enforceLimit() {
	if fm.limit <= 0 {
		return
	}
	for int(fm.openHandles.Load()) > fm.limit {
		var (
			oldest    *ManagedFile
			oldestUse = int64(math.MaxInt64)
		)
		fm.files.Range(func(_, value interface{}) bool {
			file, ok := value.(*ManagedFile)
			if !ok {
				return true
			}
			if !file.canCloseHandle() {
				return true
			}
			last := file.lastUsed.Load()
			if last == 0 {
				last = time.Now().UnixNano()
			}
			if last < oldestUse {
				oldestUse = last
				oldest = file
			}
			return true
		})
		if oldest == nil {
			break
		}
		if !oldest.closeHandleIfIdle() {
			break
		}
	}
}

func (fm *FileManager) ForceCheckpoint(opts FileCheckpointOptions) int {
	if fm == nil {
		return 0
	}
	now := time.Now().UnixNano()
	var flushed int
	fm.files.Range(func(_, value interface{}) bool {
		file, ok := value.(*ManagedFile)
		if !ok || file == nil {
			return true
		}
		if opts.IdleThreshold > 0 {
			last := file.lastUsed.Load()
			if last != 0 && now-last < opts.IdleThreshold.Nanoseconds() {
				return true
			}
		}
		if opts.DisableCache {
			file.DisableCache()
		} else {
			file.Flush()
		}
		if opts.CloseHandles {
			file.forceCloseHandle()
		}
		flushed++
		return true
	})
	return flushed
}

type sectorEntry struct {
	data       []byte
	dirty      bool
	lastAccess int64
}

type ManagedFile struct {
	path    string
	opts    ManagedFileOptions
	manager *FileManager

	file     *os.File
	handleMu sync.Mutex

	cacheMu    sync.RWMutex
	sectors    map[uint64]*sectorEntry
	maxSectors int
	sectorSize int64

	pendingMu sync.Mutex
	pending   map[uint64]struct{}

	refCount atomic.Int32
	lastUsed atomic.Int64

	cacheEnabled atomic.Bool
}

func NewManagedFile(manager *FileManager, path string, opts ManagedFileOptions) (*ManagedFile, error) {
	if opts.SectorSize <= 0 {
		opts.SectorSize = defaultSectorSize
	}
	if opts.MaxCachedSectors <= 0 {
		opts.MaxCachedSectors = defaultMaxCachedSectors
	}
	mf := &ManagedFile{
		path:       path,
		opts:       opts,
		manager:    manager,
		sectors:    make(map[uint64]*sectorEntry),
		maxSectors: opts.MaxCachedSectors,
		sectorSize: opts.SectorSize,
	}
	mf.cacheEnabled.Store(opts.CacheEnabled)
	if mf.cacheEnabled.Load() {
		if opts.FlushInterval <= 0 {
			opts.FlushInterval = 50 * time.Millisecond
		}
		mf.opts.FlushInterval = opts.FlushInterval
		mf.pending = make(map[uint64]struct{})
	}
	if manager != nil {
		manager.register(mf)
	}
	if err := mf.ensureFileExists(opts.PreallocateSize); err != nil {
		return nil, err
	}
	return mf, nil
}

func (mf *ManagedFile) ensureFileExists(preallocate int64) error {
	if _, err := os.Stat(mf.path); err == nil {
		return nil
	} else if !os.IsNotExist(err) {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(mf.path), 0755); err != nil {
		return err
	}
	file, err := os.OpenFile(mf.path, os.O_RDWR|os.O_CREATE, 0644)
	if err != nil {
		return err
	}
	if preallocate > 0 {
		if err := file.Truncate(preallocate); err != nil {
			file.Close()
			return err
		}
	}
	return file.Close()
}

func (mf *ManagedFile) beginUse() {
	mf.refCount.Add(1)
	mf.touch()
}

func (mf *ManagedFile) endUse() {
	mf.refCount.Add(-1)
}

func (mf *ManagedFile) touch() {
	mf.lastUsed.Store(time.Now().UnixNano())
}

func (mf *ManagedFile) ensureHandle() error {
	mf.handleMu.Lock()
	defer mf.handleMu.Unlock()
	if mf.file != nil {
		return nil
	}
	file, err := os.OpenFile(mf.path, os.O_RDWR|os.O_CREATE, 0644)
	if err != nil {
		return err
	}
	mf.file = file
	if mf.manager != nil {
		mf.manager.handleOpened()
	}
	return nil
}

func (mf *ManagedFile) canCloseHandle() bool {
	mf.handleMu.Lock()
	defer mf.handleMu.Unlock()
	return mf.file != nil && mf.refCount.Load() == 0
}

func (mf *ManagedFile) closeHandleIfIdle() bool {
	mf.handleMu.Lock()
	defer mf.handleMu.Unlock()
	if mf.file == nil || mf.refCount.Load() > 0 {
		return false
	}
	mf.file.Close()
	mf.file = nil
	if mf.manager != nil {
		mf.manager.handleClosed()
	}
	return true
}

func (mf *ManagedFile) forceCloseHandle() {
	mf.handleMu.Lock()
	defer mf.handleMu.Unlock()
	if mf.file == nil {
		return
	}
	mf.file.Close()
	mf.file = nil
	if mf.manager != nil {
		mf.manager.handleClosed()
	}
}

func (mf *ManagedFile) ReadAt(p []byte, off int64) (int, error) {
	if len(p) == 0 {
		return 0, nil
	}
	if err := mf.ensureHandle(); err != nil {
		return 0, err
	}
	mf.beginUse()
	defer mf.endUse()

	if !mf.cacheEnabled.Load() {
		return mf.file.ReadAt(p, off)
	}
	return mf.readWithCache(p, off)
}

func (mf *ManagedFile) readWithCache(p []byte, off int64) (int, error) {
	read := 0
	for read < len(p) {
		sectorIdx := (off + int64(read)) / mf.sectorSize
		sectorOffset := (off + int64(read)) % mf.sectorSize
		chunk := int(min64(mf.sectorSize-sectorOffset, int64(len(p)-read)))
		buf, err := mf.getSector(sectorIdx)
		if err != nil && !errors.Is(err, io.EOF) {
			return read, err
		}
		if buf == nil {
			for i := 0; i < chunk; i++ {
				p[read+i] = 0
			}
		} else {
			copy(p[read:read+chunk], buf[sectorOffset:int64(sectorOffset)+int64(chunk)])
		}
		read += chunk
		if err == io.EOF {
			break
		}
	}
	if read < len(p) {
		return read, io.EOF
	}
	return read, nil
}

func (mf *ManagedFile) WriteAt(p []byte, off int64) (int, error) {
	if len(p) == 0 {
		return 0, nil
	}
	if err := mf.ensureHandle(); err != nil {
		return 0, err
	}
	mf.beginUse()
	defer mf.endUse()

	if !mf.cacheEnabled.Load() {
		n, err := mf.file.WriteAt(p, off)
		if err == nil {
			mf.touch()
		}
		return n, err
	}
	return mf.writeWithCache(p, off)
}

func (mf *ManagedFile) writeWithCache(p []byte, off int64) (int, error) {
	written := 0
	for written < len(p) {
		sectorIdx := (off + int64(written)) / mf.sectorSize
		sectorOffset := (off + int64(written)) % mf.sectorSize
		chunk := int(min64(mf.sectorSize-sectorOffset, int64(len(p)-written)))
		buf, err := mf.ensureSector(sectorIdx)
		if err != nil {
			return written, err
		}
		copy(buf[sectorOffset:int64(sectorOffset)+int64(chunk)], p[written:written+chunk])
		mf.markDirty(sectorIdx)
		written += chunk
	}
	return written, nil
}

func (mf *ManagedFile) ensureSector(idx int64) ([]byte, error) {
	buf, err := mf.getSector(idx)
	if buf != nil || err != io.EOF {
		return buf, err
	}
	// create zero-filled sector when EOF and no data
	mf.cacheMu.Lock()
	defer mf.cacheMu.Unlock()
	entry, ok := mf.sectors[uint64(idx)]
	if !ok {
		entry = &sectorEntry{data: make([]byte, mf.sectorSize)}
		mf.sectors[uint64(idx)] = entry
	}
	entry.lastAccess = time.Now().UnixNano()
	return entry.data, nil
}

func (mf *ManagedFile) getSector(idx int64) ([]byte, error) {
	key := uint64(idx)
	mf.cacheMu.RLock()
	if entry, ok := mf.sectors[key]; ok {
		entry.lastAccess = time.Now().UnixNano()
		buf := entry.data
		mf.cacheMu.RUnlock()
		return buf, nil
	}
	mf.cacheMu.RUnlock()

	mf.cacheMu.Lock()
	defer mf.cacheMu.Unlock()
	if entry, ok := mf.sectors[key]; ok {
		entry.lastAccess = time.Now().UnixNano()
		return entry.data, nil
	}
	data := make([]byte, mf.sectorSize)
	n, err := mf.file.ReadAt(data, int64(idx)*mf.sectorSize)
	if err != nil && err != io.EOF {
		return nil, err
	}
	if n == 0 && err == io.EOF {
		return nil, io.EOF
	}
	if n < len(data) {
		for i := n; i < len(data); i++ {
			data[i] = 0
		}
	}
	entry := &sectorEntry{data: data, lastAccess: time.Now().UnixNano()}
	mf.sectors[key] = entry
	mf.evictIfNeeded()
	return entry.data, err
}

func (mf *ManagedFile) evictIfNeeded() {
	if len(mf.sectors) <= mf.maxSectors {
		return
	}
	var (
		oldestKey uint64
		oldestUse = int64(math.MaxInt64)
	)
	for key, entry := range mf.sectors {
		if entry.dirty {
			continue
		}
		if entry.lastAccess < oldestUse {
			oldestUse = entry.lastAccess
			oldestKey = key
		}
	}
	if oldestUse == int64(math.MaxInt64) {
		return
	}
	delete(mf.sectors, oldestKey)
}

func (mf *ManagedFile) markDirty(idx int64) {
	key := uint64(idx)
	mf.cacheMu.Lock()
	entry, ok := mf.sectors[key]
	if !ok {
		entry = &sectorEntry{data: make([]byte, mf.sectorSize)}
		mf.sectors[key] = entry
	}
	entry.dirty = true
	entry.lastAccess = time.Now().UnixNano()
	mf.cacheMu.Unlock()
	mf.queueFlush(key)
}

func (mf *ManagedFile) queueFlush(key uint64) {
	mf.pendingMu.Lock()
	if mf.pending == nil {
		mf.pending = make(map[uint64]struct{})
	}
	if _, exists := mf.pending[key]; exists {
		mf.pendingMu.Unlock()
		return
	}
	mf.pending[key] = struct{}{}
	mf.pendingMu.Unlock()
	if mf.manager != nil && mf.manager.enqueueFlush(mf, key) {
		return
	}
	go mf.flushSector(key)
}

func (mf *ManagedFile) flushSector(key uint64) {
	if err := mf.ensureHandle(); err != nil {
		return
	}
	mf.cacheMu.RLock()
	entry, ok := mf.sectors[key]
	dirty := ok && entry.dirty
	var buf []byte
	if dirty {
		buf = append([]byte(nil), entry.data...)
	}
	mf.cacheMu.RUnlock()
	if !dirty {
		mf.pendingMu.Lock()
		delete(mf.pending, key)
		mf.pendingMu.Unlock()
		return
	}
	_, err := mf.file.WriteAt(buf, int64(key)*mf.sectorSize)
	if err == nil {
		mf.cacheMu.Lock()
		if entry, ok := mf.sectors[key]; ok {
			entry.dirty = false
		}
		mf.cacheMu.Unlock()
	}
	mf.pendingMu.Lock()
	delete(mf.pending, key)
	mf.pendingMu.Unlock()
}

func (mf *ManagedFile) Flush() {
	if !mf.cacheEnabled.Load() {
		if mf.file != nil {
			_ = mf.file.Sync()
		}
		return
	}
	mf.cacheMu.RLock()
	dirtyKeys := make([]uint64, 0, len(mf.sectors))
	for key, entry := range mf.sectors {
		if entry.dirty {
			dirtyKeys = append(dirtyKeys, key)
		}
	}
	mf.cacheMu.RUnlock()
	for _, key := range dirtyKeys {
		mf.flushSector(key)
	}
	if mf.file != nil {
		_ = mf.file.Sync()
	}
}

func (mf *ManagedFile) Close() {
	mf.Flush()
	mf.forceCloseHandle()
}

func (mf *ManagedFile) SetCacheEnabled(enabled bool) {
	if enabled {
		if mf.cacheEnabled.Load() {
			return
		}
		mf.cacheMu.Lock()
		if mf.sectors == nil {
			mf.sectors = make(map[uint64]*sectorEntry)
		}
		mf.cacheMu.Unlock()
		mf.pendingMu.Lock()
		if mf.pending == nil {
			mf.pending = make(map[uint64]struct{})
		}
		mf.pendingMu.Unlock()
		mf.cacheEnabled.Store(true)
		mf.opts.CacheEnabled = true
		return
	}

	if !mf.cacheEnabled.Load() {
		if mf.file != nil {
			_ = mf.file.Sync()
		}
		return
	}
	mf.Flush()
	mf.cacheEnabled.Store(false)
	mf.opts.CacheEnabled = false
	mf.cacheMu.Lock()
	mf.sectors = nil
	mf.cacheMu.Unlock()
	mf.pendingMu.Lock()
	mf.pending = nil
	mf.pendingMu.Unlock()
}

func (mf *ManagedFile) DisableCache() {
	mf.SetCacheEnabled(false)
}

func min64(a, b int64) int64 {
	if a < b {
		return a
	}
	return b
}
