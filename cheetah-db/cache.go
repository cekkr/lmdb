package main

import (
	"container/list"
	"os"
	"strconv"
	"sync"
)

const (
	defaultPayloadCacheEntries = 16384
	defaultPayloadCacheBytes   = 64 << 20
)

type payloadCacheKey struct {
	size    uint32
	tableID uint32
	entryID uint16
}

type payloadCacheEntry struct {
	key     payloadCacheKey
	payload []byte
}

type payloadCache struct {
	maxEntries int
	maxBytes   int64

	mu       sync.Mutex
	order    *list.List
	entries  map[payloadCacheKey]*list.Element
	curBytes int64
}

func newPayloadCache(maxEntries int, maxBytes int64) *payloadCache {
	if maxEntries <= 0 || maxBytes <= 0 {
		return nil
	}
	return &payloadCache{
		maxEntries: maxEntries,
		maxBytes:   maxBytes,
		order:      list.New(),
		entries:    make(map[payloadCacheKey]*list.Element),
	}
}

func newPayloadCacheFromEnv() *payloadCache {
	entries := defaultPayloadCacheEntries
	maxBytes := int64(defaultPayloadCacheBytes)

	if v := os.Getenv("CHEETAH_PAYLOAD_CACHE_ENTRIES"); v != "" {
		if parsed, err := strconv.Atoi(v); err == nil && parsed > 0 {
			entries = parsed
		}
	}

	if v := os.Getenv("CHEETAH_PAYLOAD_CACHE_MB"); v != "" {
		if parsed, err := strconv.Atoi(v); err == nil && parsed > 0 {
			maxBytes = int64(parsed) << 20
		}
	}

	if v := os.Getenv("CHEETAH_PAYLOAD_CACHE_BYTES"); v != "" {
		if parsed, err := strconv.ParseInt(v, 10, 64); err == nil && parsed > 0 {
			maxBytes = parsed
		}
	}

	return newPayloadCache(entries, maxBytes)
}

func (c *payloadCache) Get(key payloadCacheKey) ([]byte, bool) {
	if c == nil {
		return nil, false
	}
	c.mu.Lock()
	defer c.mu.Unlock()

	if elem, ok := c.entries[key]; ok {
		c.order.MoveToFront(elem)
		entry := elem.Value.(*payloadCacheEntry)
		return cloneBytes(entry.payload), true
	}
	return nil, false
}

func (c *payloadCache) Add(key payloadCacheKey, payload []byte) {
	if c == nil || len(payload) == 0 {
		return
	}

	data := cloneBytes(payload)

	c.mu.Lock()
	defer c.mu.Unlock()

	if elem, ok := c.entries[key]; ok {
		existing := elem.Value.(*payloadCacheEntry)
		c.curBytes -= int64(len(existing.payload))
		existing.payload = data
		c.curBytes += int64(len(existing.payload))
		c.order.MoveToFront(elem)
	} else {
		elem := c.order.PushFront(&payloadCacheEntry{key: key, payload: data})
		c.entries[key] = elem
		c.curBytes += int64(len(data))
	}

	c.evictIfNeeded()
}

func (c *payloadCache) Invalidate(key payloadCacheKey) {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()

	if elem, ok := c.entries[key]; ok {
		c.removeElement(elem)
	}
}

func (c *payloadCache) evictIfNeeded() {
	for (c.maxEntries > 0 && len(c.entries) > c.maxEntries) || (c.maxBytes > 0 && c.curBytes > c.maxBytes) {
		elem := c.order.Back()
		if elem == nil {
			return
		}
		c.removeElement(elem)
	}
}

func (c *payloadCache) removeElement(elem *list.Element) {
	entry := elem.Value.(*payloadCacheEntry)
	c.curBytes -= int64(len(entry.payload))
	delete(c.entries, entry.key)
	c.order.Remove(elem)
}

func cloneBytes(src []byte) []byte {
	dst := make([]byte, len(src))
	copy(dst, src)
	return dst
}
