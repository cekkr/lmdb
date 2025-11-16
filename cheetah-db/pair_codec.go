package main

import (
	"errors"
	"fmt"
)

type pairBranchCodec struct {
	chunkBytes  int
	base        int
	branchCount int
}

func newPairBranchCodec(chunkBytes int) (pairBranchCodec, error) {
	if chunkBytes <= 0 {
		return pairBranchCodec{}, errors.New("chunkBytes must be positive")
	}
	if chunkBytes > 2 {
		return pairBranchCodec{}, fmt.Errorf("chunkBytes=%d exceeds supported maximum (2)", chunkBytes)
	}
	base := 1
	for i := 0; i < chunkBytes; i++ {
		base *= 256
	}
	return pairBranchCodec{
		chunkBytes:  chunkBytes,
		base:        base,
		branchCount: chunkBytes * base,
	}, nil
}

func (c pairBranchCodec) branchIndexFromChunk(chunk []byte) (uint32, error) {
	if len(chunk) == 0 || len(chunk) > c.chunkBytes {
		return 0, fmt.Errorf("invalid chunk length %d", len(chunk))
	}
	value := 0
	for _, b := range chunk {
		value = (value << 8) | int(b)
	}
	index := (len(chunk)-1)*c.base + value
	return uint32(index), nil
}

func (c pairBranchCodec) decode(index uint32) ([]byte, bool) {
	if index >= uint32(c.branchCount) {
		return nil, false
	}
	group := int(index) / c.base
	if group >= c.chunkBytes {
		return nil, false
	}
	length := group + 1
	value := int(index) % c.base
	chunk := make([]byte, length)
	for i := length - 1; i >= 0; i-- {
		chunk[i] = byte(value & 0xFF)
		value >>= 8
	}
	return chunk, true
}

func (c pairBranchCodec) walkKey(key []byte, fn func(index uint32, chunk []byte, isLast bool) error) error {
	if len(key) == 0 {
		return errors.New("empty key")
	}
	for offset := 0; offset < len(key); {
		next := offset + c.chunkBytes
		if next > len(key) {
			next = len(key)
		}
		chunk := key[offset:next]
		index, err := c.branchIndexFromChunk(chunk)
		if err != nil {
			return err
		}
		isLast := next == len(key)
		if err := fn(index, chunk, isLast); err != nil {
			return err
		}
		offset = next
	}
	return nil
}
