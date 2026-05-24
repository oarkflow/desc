package main

import (
	"encoding/json"
	"io"
	"log/slog"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestWithOCRDefaults(t *testing.T) {
	values := url.Values{"document_type": {"citizenship"}}
	got := withOCRDefaults(values, "fast", "false")

	if got.Get("accuracy_mode") != "fast" {
		t.Fatalf("accuracy_mode default = %q", got.Get("accuracy_mode"))
	}
	if got.Get("retry") != "false" {
		t.Fatalf("retry default = %q", got.Get("retry"))
	}
	if got.Get("values_only") != "true" {
		t.Fatalf("values_only default = %q", got.Get("values_only"))
	}
	if values.Get("accuracy_mode") != "" {
		t.Fatal("input values were mutated")
	}
}

func TestWithOCRDefaultsPreservesOverrides(t *testing.T) {
	values := url.Values{"accuracy_mode": {"accurate"}, "retry": {"true"}, "values_only": {"false"}}
	got := withOCRDefaults(values, "fast", "false")

	if got.Get("accuracy_mode") != "accurate" || got.Get("retry") != "true" || got.Get("values_only") != "false" {
		t.Fatalf("overrides were not preserved: %s", got.Encode())
	}
}

func TestOCRProxyRequiresAPIKeyWhenConfigured(t *testing.T) {
	gw := testGateway(t, "secret", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	req := httptest.NewRequest(http.MethodPost, "/ocr", strings.NewReader(""))
	req.Header.Set("Content-Type", "multipart/form-data; boundary=x")
	rec := httptest.NewRecorder()

	gw.routes().ServeHTTP(rec, req)

	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("status = %d, want %d", rec.Code, http.StatusUnauthorized)
	}
}

func TestOCRProxyForwardsBodyAndDefaults(t *testing.T) {
	var upstreamQuery string
	var upstreamBody string
	upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamQuery = r.URL.RawQuery
		body, _ := io.ReadAll(r.Body)
		upstreamBody = string(body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"values":{"ok":"yes"},"fields":{},"items":[],"meta":{}}`))
	})
	gw := testGateway(t, "", upstream)

	body, contentType := multipartBody(t)
	req := httptest.NewRequest(http.MethodPost, "/ocr?document_type=x", strings.NewReader(body))
	req.Header.Set("Content-Type", contentType)
	rec := httptest.NewRecorder()

	gw.routes().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	values, err := url.ParseQuery(upstreamQuery)
	if err != nil {
		t.Fatal(err)
	}
	if values.Get("accuracy_mode") != "fast" || values.Get("retry") != "false" || values.Get("document_type") != "x" {
		t.Fatalf("unexpected upstream query: %s", upstreamQuery)
	}
	if values.Get("values_only") != "true" || values.Get("fields_only") != "" {
		t.Fatalf("unexpected response shape defaults: %s", upstreamQuery)
	}
	if !strings.Contains(upstreamBody, "hello") {
		t.Fatalf("upstream body did not contain upload data")
	}
}

func TestOCRRejectsQueueFull(t *testing.T) {
	gw := testGateway(t, "", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	gw.active <- struct{}{}
	gw.metrics.activeOCR.Add(1)
	defer func() {
		<-gw.active
		gw.metrics.activeOCR.Add(-1)
	}()

	body, contentType := multipartBody(t)
	req := httptest.NewRequest(http.MethodPost, "/ocr", strings.NewReader(body))
	req.Header.Set("Content-Type", contentType)
	rec := httptest.NewRecorder()

	gw.routes().ServeHTTP(rec, req)

	if rec.Code != http.StatusTooManyRequests {
		t.Fatalf("status = %d, want %d", rec.Code, http.StatusTooManyRequests)
	}
}

func TestOCRRejectsLargeBody(t *testing.T) {
	gw := testGateway(t, "", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	gw.cfg.maxBodyBytes = 8

	body, contentType := multipartBody(t)
	req := httptest.NewRequest(http.MethodPost, "/ocr", strings.NewReader(body))
	req.Header.Set("Content-Type", contentType)
	rec := httptest.NewRecorder()

	gw.routes().ServeHTTP(rec, req)

	if rec.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("status = %d, want %d", rec.Code, http.StatusRequestEntityTooLarge)
	}
}

func TestAdminRequiresAPIKey(t *testing.T) {
	gw := testGateway(t, "secret", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	req := httptest.NewRequest(http.MethodGet, "/admin/api/document-types", nil)
	rec := httptest.NewRecorder()

	gw.routes().ServeHTTP(rec, req)

	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("status = %d, want %d", rec.Code, http.StatusUnauthorized)
	}
}

func TestRegionEditorRequiresAPIKey(t *testing.T) {
	gw := testGateway(t, "secret", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	req := httptest.NewRequest(http.MethodGet, "/region-editor", nil)
	rec := httptest.NewRecorder()

	gw.routes().ServeHTTP(rec, req)

	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("status = %d, want %d", rec.Code, http.StatusUnauthorized)
	}
}

func TestRegionEditorPageAndStaticServedWithAuth(t *testing.T) {
	gw := testGateway(t, "secret", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))

	pageReq := httptest.NewRequest(http.MethodGet, "/region-editor", nil)
	pageReq.Header.Set("X-API-Key", "secret")
	pageRec := httptest.NewRecorder()
	gw.routes().ServeHTTP(pageRec, pageReq)
	if pageRec.Code != http.StatusOK {
		t.Fatalf("page status = %d, body = %s", pageRec.Code, pageRec.Body.String())
	}
	if !strings.Contains(pageRec.Body.String(), "OCR Region Editor") {
		t.Fatalf("page did not render region editor: %s", pageRec.Body.String())
	}

	staticReq := httptest.NewRequest(http.MethodGet, "/region-editor/static/region-editor.js", nil)
	staticReq.Header.Set("X-API-Key", "secret")
	staticRec := httptest.NewRecorder()
	gw.routes().ServeHTTP(staticRec, staticReq)
	if staticRec.Code != http.StatusOK {
		t.Fatalf("static status = %d, body = %s", staticRec.Code, staticRec.Body.String())
	}
	if !strings.Contains(staticRec.Body.String(), "saveProfile") {
		t.Fatalf("static asset did not contain editor code")
	}
}

func TestAdminDocumentTypeSaveCreatesBackupAndReloads(t *testing.T) {
	var reloads int
	gw := testAdminGateway(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/admin/validate/document-type":
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`{"valid":true}`))
		case "/admin/reload":
			reloads++
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`{"status":"reloaded"}`))
		default:
			t.Fatalf("unexpected upstream path: %s", r.URL.Path)
		}
	}))
	path := gw.cfg.configDir + "/document_types/test_doc.yaml"
	if err := os.MkdirAll(gw.cfg.configDir+"/document_types", 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("document_type: test_doc\nfields: {}\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	body := `{"name":"test_doc","content":"document_type: test_doc\nfields:\n  name:\n    labels:\n    - Full Name\n"}`
	req := httptest.NewRequest(http.MethodPut, "/admin/api/document-types/test_doc", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	gw.routes().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	if reloads != 1 {
		t.Fatalf("reloads = %d, want 1", reloads)
	}
	updated, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(updated), "Full Name") {
		t.Fatalf("updated file missing content: %s", string(updated))
	}
	backups, err := filepath.Glob(gw.cfg.backupDir + "/*/config/document_types/test_doc.yaml")
	if err != nil {
		t.Fatal(err)
	}
	if len(backups) != 1 {
		t.Fatalf("backups = %v, want one backup", backups)
	}
}

func TestAdminRejectsInvalidDocumentTypeBeforeWrite(t *testing.T) {
	gw := testAdminGateway(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/admin/validate/document-type" {
			t.Fatalf("unexpected upstream path: %s", r.URL.Path)
		}
		writeJSON(w, http.StatusUnprocessableEntity, map[string]any{"valid": false, "error": "bad yaml"})
	}))
	path := gw.cfg.configDir + "/document_types/test_doc.yaml"
	if err := os.MkdirAll(gw.cfg.configDir+"/document_types", 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("document_type: test_doc\nfields: {}\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	body := `{"name":"test_doc","content":"document_type: test_doc\nfields: []\n"}`
	req := httptest.NewRequest(http.MethodPut, "/admin/api/document-types/test_doc", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	gw.routes().ServeHTTP(rec, req)

	if rec.Code != http.StatusUnprocessableEntity {
		t.Fatalf("status = %d, want %d, body = %s", rec.Code, http.StatusUnprocessableEntity, rec.Body.String())
	}
	content, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(content) != "document_type: test_doc\nfields: {}\n" {
		t.Fatalf("file changed after invalid save: %s", string(content))
	}
}

func TestAdminDeleteMovesDocumentTypeToBackup(t *testing.T) {
	gw := testAdminGateway(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/admin/reload" {
			w.WriteHeader(http.StatusOK)
			return
		}
		t.Fatalf("unexpected upstream path: %s", r.URL.Path)
	}))
	path := gw.cfg.configDir + "/document_types/delete_me.yaml"
	if err := os.MkdirAll(gw.cfg.configDir+"/document_types", 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("document_type: delete_me\nfields: {}\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	req := httptest.NewRequest(http.MethodDelete, "/admin/api/document-types/delete_me", nil)
	rec := httptest.NewRecorder()

	gw.routes().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("deleted file still exists or stat failed unexpectedly: %v", err)
	}
	backups, err := filepath.Glob(gw.cfg.backupDir + "/*/config/document_types/delete_me.yaml")
	if err != nil {
		t.Fatal(err)
	}
	if len(backups) != 1 {
		t.Fatalf("backups = %v, want one backup", backups)
	}
}

func TestAdminPreviewForwardsMultipartUpload(t *testing.T) {
	var gotContentType string
	var gotQuery string
	var gotFileContentType string
	var gotFile string
	gw := testAdminGateway(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/ocr" {
			t.Fatalf("unexpected upstream path: %s", r.URL.Path)
		}
		gotContentType = r.Header.Get("Content-Type")
		gotQuery = r.URL.RawQuery
		if err := r.ParseMultipartForm(1024 * 1024); err != nil {
			t.Fatal(err)
		}
		file, header, err := r.FormFile("file")
		if err != nil {
			t.Fatal(err)
		}
		defer file.Close()
		gotFileContentType = header.Header.Get("Content-Type")
		body, _ := io.ReadAll(file)
		gotFile = string(body)
		writeJSON(w, http.StatusOK, map[string]any{"items": []any{}, "width": 1, "height": 1})
	}))

	body, contentType := multipartBody(t)
	req := httptest.NewRequest(http.MethodPost, "/admin/api/preview", strings.NewReader(body))
	req.Header.Set("Content-Type", contentType)
	rec := httptest.NewRecorder()

	gw.routes().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	if !strings.HasPrefix(gotContentType, "multipart/form-data") {
		t.Fatalf("content-type = %q", gotContentType)
	}
	values, err := url.ParseQuery(gotQuery)
	if err != nil {
		t.Fatal(err)
	}
	if values.Get("accuracy_mode") != "fast" || values.Get("retry") != "false" {
		t.Fatalf("query defaults missing: %s", gotQuery)
	}
	if gotFile != "hello" {
		t.Fatalf("file = %q, want hello", gotFile)
	}
	if gotFileContentType == "" {
		t.Fatal("file content type was not preserved")
	}
	var payload map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &payload); err != nil {
		t.Fatal(err)
	}
}

func TestAdminParseAndRenderDocumentTypeProxy(t *testing.T) {
	var paths []string
	gw := testAdminGateway(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		paths = append(paths, r.URL.Path)
		body, _ := io.ReadAll(r.Body)
		if len(body) == 0 {
			t.Fatal("expected request body")
		}
		writeJSON(w, http.StatusOK, map[string]any{"ok": true})
	}))

	parseReq := httptest.NewRequest(http.MethodPost, "/admin/api/parse/document-type", strings.NewReader("document_type: x\nfields: {}\n"))
	parseReq.Header.Set("Content-Type", "application/x-yaml")
	parseRec := httptest.NewRecorder()
	gw.routes().ServeHTTP(parseRec, parseReq)
	if parseRec.Code != http.StatusOK {
		t.Fatalf("parse status = %d, body = %s", parseRec.Code, parseRec.Body.String())
	}

	renderReq := httptest.NewRequest(http.MethodPost, "/admin/api/render/document-type", strings.NewReader(`{"document_type":"x","profile":{"fields":{}}}`))
	renderReq.Header.Set("Content-Type", "application/json")
	renderRec := httptest.NewRecorder()
	gw.routes().ServeHTTP(renderRec, renderReq)
	if renderRec.Code != http.StatusOK {
		t.Fatalf("render status = %d, body = %s", renderRec.Code, renderRec.Body.String())
	}

	got := strings.Join(paths, ",")
	want := "/admin/parse/document-type,/admin/render/document-type"
	if got != want {
		t.Fatalf("paths = %s, want %s", got, want)
	}
}

func testGateway(t *testing.T, apiKey string, upstream http.Handler) *gateway {
	t.Helper()
	server := httptest.NewServer(upstream)
	t.Cleanup(server.Close)
	parsed, err := url.Parse(server.URL)
	if err != nil {
		t.Fatal(err)
	}
	return newGateway(config{
		upstreamURLs:    []*url.URL{parsed},
		apiKey:          apiKey,
		maxBodyBytes:    1024 * 1024,
		maxActive:       1,
		maxQueue:        0,
		upstreamTimeout: time.Second,
		defaultAccuracy: "fast",
		defaultRetry:    "false",
	}, nilLogger())
}

func testAdminGateway(t *testing.T, upstream http.Handler) *gateway {
	t.Helper()
	gw := testGateway(t, "", upstream)
	root := t.TempDir()
	gw.cfg.configDir = filepath.Join(root, "config")
	gw.cfg.dataDir = filepath.Join(root, "data")
	gw.cfg.backupDir = filepath.Join(root, "backups")
	return gw
}

func multipartBody(t *testing.T) (string, string) {
	t.Helper()
	var b strings.Builder
	writer := multipart.NewWriter(&b)
	part, err := writer.CreateFormFile("file", "test.txt")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := part.Write([]byte("hello")); err != nil {
		t.Fatal(err)
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	return b.String(), writer.FormDataContentType()
}

func nilLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(io.Discard, nil))
}
