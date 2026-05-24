package main

import (
	"bytes"
	"context"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"mime"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

type config struct {
	listenAddr       string
	upstreamURLs     []*url.URL
	apiKey           string
	configDir        string
	dataDir          string
	backupDir        string
	maxBodyBytes     int64
	maxActive        int
	maxQueue         int
	upstreamTimeout  time.Duration
	shutdownTimeout  time.Duration
	defaultAccuracy  string
	defaultRetry     string
	readHeaderTimout time.Duration
}

type gateway struct {
	cfg        config
	client     *http.Client
	active     chan struct{}
	queue      chan struct{}
	nextWorker atomic.Uint64
	metrics    *metrics
	logger     *slog.Logger
}

type metrics struct {
	requestsTotal        atomic.Uint64
	ocrRequestsTotal     atomic.Uint64
	ocrSuccessTotal      atomic.Uint64
	ocrErrorsTotal       atomic.Uint64
	authFailuresTotal    atomic.Uint64
	queueRejectedTotal   atomic.Uint64
	requestTooLargeTotal atomic.Uint64
	upstreamErrorsTotal  atomic.Uint64
	activeOCR            atomic.Int64
	queuedOCR            atomic.Int64
	ocrLatencyMS         atomic.Uint64
	queueWaitMS          atomic.Uint64
	mu                   sync.Mutex
	statusCounts         map[int]uint64
}

func newMetrics() *metrics {
	return &metrics{statusCounts: make(map[int]uint64)}
}

func main() {
	cfg, err := loadConfig()
	if err != nil {
		slog.Error("invalid configuration", "error", err)
		os.Exit(1)
	}

	logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo}))
	gw := newGateway(cfg, logger)
	server := &http.Server{
		Addr:              cfg.listenAddr,
		Handler:           gw.routes(),
		ReadHeaderTimeout: cfg.readHeaderTimout,
	}

	logger.Info("starting gateway", "addr", cfg.listenAddr, "upstreams", len(cfg.upstreamURLs), "max_active", cfg.maxActive, "max_queue", cfg.maxQueue)
	if err := server.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		logger.Error("gateway stopped", "error", err)
		os.Exit(1)
	}
}

func loadConfig() (config, error) {
	upstreams, err := parseUpstreams(envString("GATEWAY_OCR_UPSTREAMS", envString("GATEWAY_OCR_UPSTREAM", "http://ocr:8000")))
	if err != nil {
		return config{}, err
	}

	maxFileMB := envInt("OCR_MAX_FILE_MB", 15)
	maxActive := envInt("GATEWAY_MAX_ACTIVE", envInt("OCR_WORKERS", 1))
	if maxActive < 1 {
		maxActive = 1
	}
	maxQueue := envInt("GATEWAY_MAX_QUEUE", maxActive*4)
	if maxQueue < 0 {
		maxQueue = 0
	}

	return config{
		listenAddr:       ":" + envString("GATEWAY_PORT", "8000"),
		upstreamURLs:     upstreams,
		apiKey:           os.Getenv("GATEWAY_API_KEY"),
		configDir:        envString("GATEWAY_CONFIG_DIR", "config"),
		dataDir:          envString("GATEWAY_DATA_DIR", "data"),
		backupDir:        envString("GATEWAY_BACKUP_DIR", ".gateway_backups"),
		maxBodyBytes:     int64(maxFileMB) * 1024 * 1024,
		maxActive:        maxActive,
		maxQueue:         maxQueue,
		upstreamTimeout:  time.Duration(envInt("GATEWAY_UPSTREAM_TIMEOUT_SECONDS", 90)) * time.Second,
		shutdownTimeout:  time.Duration(envInt("GATEWAY_SHUTDOWN_TIMEOUT_SECONDS", 10)) * time.Second,
		defaultAccuracy:  envString("GATEWAY_DEFAULT_ACCURACY_MODE", "fast"),
		defaultRetry:     envString("GATEWAY_DEFAULT_RETRY", "false"),
		readHeaderTimout: time.Duration(envInt("GATEWAY_READ_HEADER_TIMEOUT_SECONDS", 10)) * time.Second,
	}, nil
}

func parseUpstreams(raw string) ([]*url.URL, error) {
	parts := strings.Split(raw, ",")
	upstreams := make([]*url.URL, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		parsed, err := url.Parse(part)
		if err != nil {
			return nil, fmt.Errorf("parse upstream %q: %w", part, err)
		}
		if parsed.Scheme == "" || parsed.Host == "" {
			return nil, fmt.Errorf("upstream must include scheme and host: %q", part)
		}
		upstreams = append(upstreams, parsed)
	}
	if len(upstreams) == 0 {
		return nil, errors.New("at least one OCR upstream is required")
	}
	return upstreams, nil
}

func newGateway(cfg config, logger *slog.Logger) *gateway {
	return &gateway{
		cfg:     cfg,
		client:  &http.Client{Timeout: cfg.upstreamTimeout},
		active:  make(chan struct{}, cfg.maxActive),
		queue:   make(chan struct{}, cfg.maxQueue),
		metrics: newMetrics(),
		logger:  logger,
	}
}

func (g *gateway) routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", g.health)
	mux.HandleFunc("GET /metrics", g.metricsEndpoint)
	mux.HandleFunc("POST /ocr", g.ocr)
	g.registerAdminRoutes(mux)
	mux.HandleFunc("/", g.notFound)
	return g.withMetrics(mux)
}

func (g *gateway) withMetrics(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		g.metrics.requestsTotal.Add(1)
		next.ServeHTTP(w, r)
	})
}

func (g *gateway) health(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func (g *gateway) metricsEndpoint(w http.ResponseWriter, r *http.Request) {
	if !g.authorized(r) {
		g.metrics.authFailuresTotal.Add(1)
		writeError(w, http.StatusUnauthorized, "unauthorized")
		return
	}

	w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
	snapshot := g.metricsSnapshot()
	_, _ = io.WriteString(w, snapshot)
}

func (g *gateway) ocr(w http.ResponseWriter, r *http.Request) {
	started := time.Now()
	g.metrics.ocrRequestsTotal.Add(1)

	if !g.authorized(r) {
		g.metrics.authFailuresTotal.Add(1)
		writeError(w, http.StatusUnauthorized, "unauthorized")
		g.recordStatus(http.StatusUnauthorized)
		return
	}

	if !isMultipart(r.Header.Get("Content-Type")) {
		writeError(w, http.StatusBadRequest, "multipart form upload required")
		g.recordStatus(http.StatusBadRequest)
		return
	}

	release, queueWait, ok := g.acquireOCR()
	if !ok {
		g.metrics.queueRejectedTotal.Add(1)
		writeError(w, http.StatusTooManyRequests, "ocr queue is full")
		g.recordStatus(http.StatusTooManyRequests)
		return
	}
	defer release()
	g.metrics.queueWaitMS.Add(uint64(queueWait.Milliseconds()))

	body, err := readLimited(r.Body, g.cfg.maxBodyBytes)
	if err != nil {
		if errors.Is(err, errBodyTooLarge) {
			g.metrics.requestTooLargeTotal.Add(1)
			writeError(w, http.StatusRequestEntityTooLarge, "request body too large")
			g.recordStatus(http.StatusRequestEntityTooLarge)
			return
		}
		writeError(w, http.StatusBadRequest, "failed to read request body")
		g.recordStatus(http.StatusBadRequest)
		return
	}

	upstreamReq, err := http.NewRequestWithContext(r.Context(), http.MethodPost, g.upstreamURL(r).String(), bytes.NewReader(body))
	if err != nil {
		writeError(w, http.StatusBadGateway, "failed to create upstream request")
		g.recordStatus(http.StatusBadGateway)
		return
	}
	copyProxyHeaders(upstreamReq.Header, r.Header)
	upstreamReq.ContentLength = int64(len(body))

	resp, err := g.client.Do(upstreamReq)
	if err != nil {
		status := http.StatusBadGateway
		if errors.Is(err, context.DeadlineExceeded) || strings.Contains(err.Error(), "Client.Timeout") {
			status = http.StatusGatewayTimeout
		}
		g.metrics.upstreamErrorsTotal.Add(1)
		writeError(w, status, "ocr upstream unavailable")
		g.recordStatus(status)
		return
	}
	defer resp.Body.Close()

	copyResponseHeaders(w.Header(), resp.Header)
	w.WriteHeader(resp.StatusCode)
	if _, err := io.Copy(w, resp.Body); err != nil {
		g.logger.Warn("copy upstream response failed", "error", err)
	}

	g.recordStatus(resp.StatusCode)
	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		g.metrics.ocrSuccessTotal.Add(1)
	} else {
		g.metrics.ocrErrorsTotal.Add(1)
		if resp.StatusCode >= 500 {
			g.metrics.upstreamErrorsTotal.Add(1)
		}
	}
	g.metrics.ocrLatencyMS.Add(uint64(time.Since(started).Milliseconds()))
}

func (g *gateway) notFound(w http.ResponseWriter, _ *http.Request) {
	writeError(w, http.StatusNotFound, "not found")
}

func (g *gateway) authorized(r *http.Request) bool {
	if g.cfg.apiKey == "" {
		return true
	}
	got := r.Header.Get("X-API-Key")
	if got == "" {
		got = strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
	}
	if got == "" {
		if cookie, err := r.Cookie("kyc_gateway_api_key"); err == nil {
			got = cookie.Value
		}
	}
	if got == "" {
		got = r.URL.Query().Get("api_key")
	}
	return subtle.ConstantTimeCompare([]byte(got), []byte(g.cfg.apiKey)) == 1
}

func (g *gateway) acquireOCR() (func(), time.Duration, bool) {
	queuedAt := time.Now()
	if cap(g.queue) == 0 {
		select {
		case g.active <- struct{}{}:
			g.metrics.activeOCR.Add(1)
			return func() {
				<-g.active
				g.metrics.activeOCR.Add(-1)
			}, time.Since(queuedAt), true
		default:
			return nil, 0, false
		}
	}

	if cap(g.queue) > 0 {
		select {
		case g.queue <- struct{}{}:
			g.metrics.queuedOCR.Add(1)
			defer func() {
				<-g.queue
				g.metrics.queuedOCR.Add(-1)
			}()
		default:
			return nil, 0, false
		}
	}

	g.active <- struct{}{}
	g.metrics.activeOCR.Add(1)
	return func() {
		<-g.active
		g.metrics.activeOCR.Add(-1)
	}, time.Since(queuedAt), true
}

func (g *gateway) upstreamURL(r *http.Request) *url.URL {
	index := (g.nextWorker.Add(1) - 1) % uint64(len(g.cfg.upstreamURLs))
	base := g.cfg.upstreamURLs[index]
	out := *base
	out.Path = joinPath(base.Path, "/ocr")
	out.RawQuery = withOCRDefaults(r.URL.Query(), g.cfg.defaultAccuracy, g.cfg.defaultRetry).Encode()
	return &out
}

func withOCRDefaults(values url.Values, accuracyMode, retry string) url.Values {
	out := make(url.Values, len(values)+2)
	for key, vals := range values {
		out[key] = append([]string(nil), vals...)
	}
	if out.Get("accuracy_mode") == "" {
		out.Set("accuracy_mode", accuracyMode)
	}
	if out.Get("retry") == "" {
		out.Set("retry", retry)
	}
	return out
}

func joinPath(base, suffix string) string {
	base = strings.TrimRight(base, "/")
	if base == "" {
		return suffix
	}
	return base + suffix
}

func (g *gateway) recordStatus(status int) {
	g.metrics.mu.Lock()
	defer g.metrics.mu.Unlock()
	g.metrics.statusCounts[status]++
}

func (g *gateway) metricsSnapshot() string {
	var b strings.Builder
	writeMetric := func(name, help, metricType string, value any) {
		fmt.Fprintf(&b, "# HELP %s %s\n", name, help)
		fmt.Fprintf(&b, "# TYPE %s %s\n", name, metricType)
		fmt.Fprintf(&b, "%s %v\n", name, value)
	}

	writeMetric("kyc_gateway_requests_total", "Total HTTP requests received by the gateway.", "counter", g.metrics.requestsTotal.Load())
	writeMetric("kyc_gateway_ocr_requests_total", "Total OCR proxy requests received.", "counter", g.metrics.ocrRequestsTotal.Load())
	writeMetric("kyc_gateway_ocr_success_total", "Total successful OCR proxy responses.", "counter", g.metrics.ocrSuccessTotal.Load())
	writeMetric("kyc_gateway_ocr_errors_total", "Total unsuccessful OCR proxy responses.", "counter", g.metrics.ocrErrorsTotal.Load())
	writeMetric("kyc_gateway_auth_failures_total", "Total authentication failures.", "counter", g.metrics.authFailuresTotal.Load())
	writeMetric("kyc_gateway_queue_rejected_total", "Total OCR requests rejected because queue was full.", "counter", g.metrics.queueRejectedTotal.Load())
	writeMetric("kyc_gateway_request_too_large_total", "Total requests rejected because body exceeded the limit.", "counter", g.metrics.requestTooLargeTotal.Load())
	writeMetric("kyc_gateway_upstream_errors_total", "Total upstream transport errors or 5xx responses.", "counter", g.metrics.upstreamErrorsTotal.Load())
	writeMetric("kyc_gateway_active_ocr", "Current active OCR upstream requests.", "gauge", g.metrics.activeOCR.Load())
	writeMetric("kyc_gateway_queued_ocr", "Current queued OCR requests.", "gauge", g.metrics.queuedOCR.Load())
	writeMetric("kyc_gateway_ocr_latency_ms_total", "Cumulative OCR request latency in milliseconds.", "counter", g.metrics.ocrLatencyMS.Load())
	writeMetric("kyc_gateway_queue_wait_ms_total", "Cumulative OCR queue wait in milliseconds.", "counter", g.metrics.queueWaitMS.Load())

	g.metrics.mu.Lock()
	defer g.metrics.mu.Unlock()
	for status, count := range g.metrics.statusCounts {
		fmt.Fprintf(&b, "kyc_gateway_responses_total{status=\"%d\"} %d\n", status, count)
	}
	return b.String()
}

var errBodyTooLarge = errors.New("body too large")

func readLimited(r io.ReadCloser, limit int64) ([]byte, error) {
	defer r.Close()
	var buf bytes.Buffer
	_, err := io.Copy(&buf, io.LimitReader(r, limit+1))
	if err != nil {
		return nil, err
	}
	if int64(buf.Len()) > limit {
		return nil, errBodyTooLarge
	}
	return buf.Bytes(), nil
}

func isMultipart(contentType string) bool {
	mediaType, _, err := mime.ParseMediaType(contentType)
	return err == nil && strings.EqualFold(mediaType, "multipart/form-data")
}

func copyProxyHeaders(dst, src http.Header) {
	for _, header := range []string{"Content-Type", "Accept", "User-Agent"} {
		if value := src.Get(header); value != "" {
			dst.Set(header, value)
		}
	}
}

func copyResponseHeaders(dst, src http.Header) {
	for key, values := range src {
		if strings.EqualFold(key, "Connection") || strings.EqualFold(key, "Transfer-Encoding") {
			continue
		}
		for _, value := range values {
			dst.Add(key, value)
		}
	}
}

func writeError(w http.ResponseWriter, status int, message string) {
	writeJSON(w, status, map[string]string{"error": message})
}

func writeJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(payload)
}

func envString(key, fallback string) string {
	if value := strings.TrimSpace(os.Getenv(key)); value != "" {
		return value
	}
	return fallback
}

func envInt(key string, fallback int) int {
	value := strings.TrimSpace(os.Getenv(key))
	if value == "" {
		return fallback
	}
	parsed, err := strconv.Atoi(value)
	if err != nil {
		return fallback
	}
	return parsed
}
