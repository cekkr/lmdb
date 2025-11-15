package main

import (
	"bufio"
	"errors"
	"math"
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
)

type ResourceSnapshot struct {
	Timestamp           time.Time
	LogicalCores        int
	Gomaxprocs          int
	Goroutines          int
	ProcessCPUPercent   float64
	SystemCPUPercent    float64
	ProcessCPUTime      time.Duration
	ProcessCPUSupported bool
	SystemCPUSupported  bool
	IOSupported         bool
	IOReadBytes         uint64
	IOWriteBytes        uint64
	IOReadRate          float64
	IOWriteRate         float64
	MemAllocBytes       uint64
	MemSysBytes         uint64
}

type ResourceMonitor struct {
	interval        time.Duration
	stopCh          chan struct{}
	stopOnce        sync.Once
	mu              sync.RWMutex
	snapshot        ResourceSnapshot
	lastProcCPU     time.Duration
	hasProcSample   bool
	lastSystemCPU   systemCPUSample
	hasSystemSample bool
	lastIOSample    ioSample
	hasIOSample     bool
}

type systemCPUSample struct {
	Idle  uint64
	Total uint64
}

type ioSample struct {
	ReadBytes  uint64
	WriteBytes uint64
}

func NewResourceMonitor(interval time.Duration) *ResourceMonitor {
	if interval <= 0 {
		interval = time.Second
	}
	rm := &ResourceMonitor{
		interval: interval,
		stopCh:   make(chan struct{}),
	}
	rm.takeSample(time.Now())
	go rm.loop()
	return rm
}

func (rm *ResourceMonitor) loop() {
	ticker := time.NewTicker(rm.interval)
	defer ticker.Stop()
	for {
		select {
		case now := <-ticker.C:
			rm.takeSample(now)
		case <-rm.stopCh:
			return
		}
	}
}

func (rm *ResourceMonitor) Stop() {
	rm.stopOnce.Do(func() {
		close(rm.stopCh)
	})
}

func (rm *ResourceMonitor) Snapshot() ResourceSnapshot {
	rm.mu.RLock()
	defer rm.mu.RUnlock()
	return rm.snapshot
}

func (rm *ResourceMonitor) RecommendedWorkers(pending int) int {
	if pending <= 1 {
		if pending == 0 {
			return 0
		}
		return 1
	}
	rm.mu.RLock()
	snap := rm.snapshot
	rm.mu.RUnlock()

	maxWorkers := snap.Gomaxprocs
	if maxWorkers < 1 {
		maxWorkers = runtime.GOMAXPROCS(0)
	}
	if maxWorkers < 1 {
		maxWorkers = runtime.NumCPU()
	}
	if maxWorkers < 1 {
		maxWorkers = 1
	}

	scale := 1.0
	switch {
	case snap.ProcessCPUPercent >= 80 || snap.SystemCPUPercent >= 85:
		scale = 0.5
	case snap.ProcessCPUPercent >= 60 || snap.SystemCPUPercent >= 70:
		scale = 0.75
	}
	recommended := int(math.Max(1, math.Floor(float64(maxWorkers)*scale)))
	if recommended > pending {
		recommended = pending
	}
	return recommended
}

func (rm *ResourceMonitor) takeSample(now time.Time) {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	logicalCores := runtime.NumCPU()
	gomaxprocs := runtime.GOMAXPROCS(0)
	goroutines := runtime.NumGoroutine()

	procCPUTime, procErr := getProcessCPUTime()
	systemSample, systemErr := readSystemCPUSample()
	ioSample, ioErr := readProcSelfIO()

	rm.mu.Lock()
	defer rm.mu.Unlock()

	prevSnapshot := rm.snapshot
	elapsed := now.Sub(prevSnapshot.Timestamp)

	snapshot := ResourceSnapshot{
		Timestamp:     now,
		LogicalCores:  logicalCores,
		Gomaxprocs:    gomaxprocs,
		Goroutines:    goroutines,
		MemAllocBytes: m.Alloc,
		MemSysBytes:   m.Sys,
	}

	if procErr == nil {
		snapshot.ProcessCPUSupported = true
		snapshot.ProcessCPUTime = procCPUTime
		if rm.hasProcSample && elapsed > 0 {
			procDelta := procCPUTime - rm.lastProcCPU
			if procDelta > 0 {
				cores := float64(logicalCores)
				if cores == 0 {
					cores = 1
				}
				snapshot.ProcessCPUPercent = (float64(procDelta) / float64(elapsed)) * 100 / cores
			}
		}
		rm.lastProcCPU = procCPUTime
		rm.hasProcSample = true
	}

	if systemErr == nil {
		snapshot.SystemCPUSupported = true
		if rm.hasSystemSample {
			if deltaTotal := systemSample.Total - rm.lastSystemCPU.Total; deltaTotal > 0 {
				deltaIdle := systemSample.Idle - rm.lastSystemCPU.Idle
				usage := 1 - float64(deltaIdle)/float64(deltaTotal)
				if usage < 0 {
					usage = 0
				}
				if usage > 1 {
					usage = 1
				}
				snapshot.SystemCPUPercent = usage * 100
			}
		}
		rm.lastSystemCPU = systemSample
		rm.hasSystemSample = true
	}

	if ioErr == nil {
		snapshot.IOSupported = true
		snapshot.IOReadBytes = ioSample.ReadBytes
		snapshot.IOWriteBytes = ioSample.WriteBytes
		if rm.hasIOSample && elapsed > 0 {
			readDelta := ioSample.ReadBytes - rm.lastIOSample.ReadBytes
			writeDelta := ioSample.WriteBytes - rm.lastIOSample.WriteBytes
			elapsedSeconds := elapsed.Seconds()
			snapshot.IOReadRate = float64(readDelta) / elapsedSeconds
			snapshot.IOWriteRate = float64(writeDelta) / elapsedSeconds
		}
		rm.lastIOSample = ioSample
		rm.hasIOSample = true
	}

	rm.snapshot = snapshot
}

func getProcessCPUTime() (time.Duration, error) {
	var usage syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &usage); err != nil {
		return 0, err
	}
	user := time.Duration(usage.Utime.Sec)*time.Second + time.Duration(usage.Utime.Usec)*time.Microsecond
	system := time.Duration(usage.Stime.Sec)*time.Second + time.Duration(usage.Stime.Usec)*time.Microsecond
	return user + system, nil
}

func readSystemCPUSample() (systemCPUSample, error) {
	file, err := os.Open("/proc/stat")
	if err != nil {
		return systemCPUSample{}, err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "cpu ") {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 5 {
			return systemCPUSample{}, errors.New("invalid cpu line")
		}
		var total uint64
		for _, field := range fields[1:] {
			value, err := strconv.ParseUint(field, 10, 64)
			if err != nil {
				return systemCPUSample{}, err
			}
			total += value
		}
		idle, err := strconv.ParseUint(fields[4], 10, 64)
		if err != nil {
			return systemCPUSample{}, err
		}
		return systemCPUSample{Idle: idle, Total: total}, nil
	}
	if err := scanner.Err(); err != nil {
		return systemCPUSample{}, err
	}
	return systemCPUSample{}, errors.New("cpu line not found")
}

func readProcSelfIO() (ioSample, error) {
	data, err := os.ReadFile("/proc/self/io")
	if err != nil {
		return ioSample{}, err
	}
	var sample ioSample
	scanner := bufio.NewScanner(strings.NewReader(string(data)))
	for scanner.Scan() {
		line := scanner.Text()
		switch {
		case strings.HasPrefix(line, "read_bytes:"):
			value, err := strconv.ParseUint(strings.TrimSpace(strings.TrimPrefix(line, "read_bytes:")), 10, 64)
			if err == nil {
				sample.ReadBytes = value
			}
		case strings.HasPrefix(line, "write_bytes:"):
			value, err := strconv.ParseUint(strings.TrimSpace(strings.TrimPrefix(line, "write_bytes:")), 10, 64)
			if err == nil {
				sample.WriteBytes = value
			}
		}
	}
	if err := scanner.Err(); err != nil {
		return ioSample{}, err
	}
	return sample, nil
}
