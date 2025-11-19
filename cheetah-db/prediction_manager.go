package main

import (
	"fmt"
	"path/filepath"
	"regexp"
	"sync"
)

var tableNameSanitizer = regexp.MustCompile(`[^a-zA-Z0-9_\-]`)

type PredictionManager struct {
	basePath string
	mu       sync.Mutex
	tables   map[string]*PredictionTable
}

func newPredictionManager(basePath string) *PredictionManager {
	return &PredictionManager{
		basePath: basePath,
		tables:   make(map[string]*PredictionTable),
	}
}

func (pm *PredictionManager) sanitizeTableName(raw string) string {
	name := raw
	if name == "" {
		name = "default"
	}
	name = tableNameSanitizer.ReplaceAllString(name, "_")
	if name == "" {
		name = "default"
	}
	if len(name) > 64 {
		name = name[:64]
	}
	return name
}

func (pm *PredictionManager) Get(table string) (*PredictionTable, error) {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	if pm.tables == nil {
		pm.tables = make(map[string]*PredictionTable)
	}
	name := pm.sanitizeTableName(table)
	if existing, ok := pm.tables[name]; ok {
		return existing, nil
	}
	path := filepath.Join(pm.basePath, fmt.Sprintf("prediction_%s.json", name))
	pt, err := newPredictionTable(path, name)
	if err != nil {
		return nil, err
	}
	pm.tables[name] = pt
	return pt, nil
}

func (pm *PredictionManager) Close() {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	for name, table := range pm.tables {
		if err := table.Close(); err != nil {
			logErrorf("failed closing prediction table %s: %v", name, err)
		}
	}
	pm.tables = nil
}
