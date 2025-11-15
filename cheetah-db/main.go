// main.go
package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"os/signal"
	"runtime"
	"strings"
	"syscall"
	"time"
)

const (
	ListenAddr    = "0.0.0.0:4455"
	DataDir       = "cheetah_data"
	DefaultDbName = "default"
)

func main() {
	monitor := NewResourceMonitor(2 * time.Second)
	defer monitor.Stop()
	coreSnapshot := monitor.Snapshot()
	logInfof("Detected %d logical CPU cores (GOMAXPROCS=%d)", coreSnapshot.LogicalCores, runtime.GOMAXPROCS(0))

	// Inizializza l'engine del database
	engine, err := NewEngine(DataDir, monitor)
	if err != nil {
		log.Fatalf("FATAL: Failed to start engine: %v", err)
	}
	defer engine.Close() // Assicura che tutti i DB siano chiusi all'uscita

	// Avvia il server TCP in una goroutine separata
	server := NewTCPServer(ListenAddr, engine)
	go func() {
		if err := server.Start(); err != nil {
			logErrorf("TCP Server failed: %v", err)
		}
	}()

	// Gestisce la chiusura pulita su segnali come Ctrl+C
	setupGracefulShutdown(engine, monitor)

	if os.Getenv("CHEETAH_HEADLESS") == "1" {
		logInfof("CheetahDB headless mode active. CLI disabled; press Ctrl+C to stop the server.")
		select {}
	}

	// Avvia l'interfaccia a riga di comando (CLI)
	runCLI(engine)
}

// runCLI gestisce l'input dal terminale per i comandi locali
func runCLI(engine *Engine) {
	logInfof("CheetahDB Console Interface is running. Type 'EXIT' to quit.")
	currentDB, err := engine.GetDatabase(DefaultDbName)
	if err != nil {
		log.Fatalf("FATAL: Failed to load default database for CLI: %v", err)
		return
	}

	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Printf("[%s]> ", currentDB.Path())
		if !scanner.Scan() {
			break
		}

		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		if strings.ToUpper(line) == "EXIT" {
			break
		}

		var response string
		// Logica per cambiare DB nella CLI
		parts := strings.SplitN(line, " ", 2)
		if strings.ToUpper(parts[0]) == "DATABASE" && len(parts) > 1 {
			dbName := strings.TrimSpace(parts[1])
			newDB, err := engine.GetDatabase(dbName)
			if err != nil {
				response = fmt.Sprintf("ERROR,cannot_load_db:%v", err)
			} else {
				currentDB = newDB
				response = fmt.Sprintf("SUCCESS,database_changed_to_%s", dbName)
			}
		} else {
			// Esegue tutti gli altri comandi sul DB corrente
			response, err = currentDB.ExecuteCommand(line)
			if err != nil {
				response = fmt.Sprintf("ERROR,internal_error:%v", err)
			}
		}
		fmt.Println(response)
	}
}

// setupGracefulShutdown attende un segnale di interruzione per chiudere le risorse
func setupGracefulShutdown(engine *Engine, monitor *ResourceMonitor) {
	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-c
		logInfof("Shutdown signal received. Closing resources...")
		engine.Close()
		if monitor != nil {
			monitor.Stop()
		}
		os.Exit(0)
	}()
}
