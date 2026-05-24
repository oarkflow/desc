package main

import (
	"bytes"
	"context"
	"embed"
	"encoding/json"
	"errors"
	"fmt"
	"html/template"
	"io"
	"mime"
	"mime/multipart"
	"net/http"
	"net/textproto"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"
)

//go:embed templates/* static/*
var adminFS embed.FS

var safeNamePattern = regexp.MustCompile(`^[A-Za-z0-9._-]+$`)

type adminPageData struct {
	APIKeyConfigured bool
	DefaultAPIKey    string
}

type documentTypeSummary struct {
	ID       string `json:"id"`
	File     string `json:"file"`
	Bytes    int64  `json:"bytes"`
	Modified string `json:"modified"`
}

type filePayload struct {
	Name    string `json:"name"`
	Content string `json:"content"`
}

type duplicatePayload struct {
	ID string `json:"id"`
}

type adminConfigResponse struct {
	DocumentTypes []documentTypeSummary `json:"document_types"`
	DataFiles     []documentTypeSummary `json:"data_files"`
	Profiles      filePayload           `json:"profiles"`
}

func (g *gateway) registerAdminRoutes(mux *http.ServeMux) {
	mux.HandleFunc("GET /admin", g.adminPage)
	mux.HandleFunc("GET /admin/", g.adminPage)
	mux.HandleFunc("GET /admin/static/{name}", g.adminStatic)
	mux.HandleFunc("GET /admin/api/config", g.adminAPI(g.adminConfig))
	mux.HandleFunc("GET /admin/api/document-types", g.adminAPI(g.adminListDocumentTypes))
	mux.HandleFunc("POST /admin/api/document-types", g.adminAPI(g.adminCreateDocumentType))
	mux.HandleFunc("GET /admin/api/document-types/{id}", g.adminAPI(g.adminGetDocumentType))
	mux.HandleFunc("PUT /admin/api/document-types/{id}", g.adminAPI(g.adminUpdateDocumentType))
	mux.HandleFunc("DELETE /admin/api/document-types/{id}", g.adminAPI(g.adminDeleteDocumentType))
	mux.HandleFunc("POST /admin/api/document-types/{id}/duplicate", g.adminAPI(g.adminDuplicateDocumentType))
	mux.HandleFunc("POST /admin/api/parse/document-type", g.adminAPI(g.adminParseDocumentType))
	mux.HandleFunc("POST /admin/api/render/document-type", g.adminAPI(g.adminRenderDocumentType))
	mux.HandleFunc("GET /admin/api/profiles", g.adminAPI(g.adminGetProfiles))
	mux.HandleFunc("PUT /admin/api/profiles", g.adminAPI(g.adminUpdateProfiles))
	mux.HandleFunc("GET /admin/api/data", g.adminAPI(g.adminListDataFiles))
	mux.HandleFunc("GET /admin/api/data/{name}", g.adminAPI(g.adminGetDataFile))
	mux.HandleFunc("PUT /admin/api/data/{name}", g.adminAPI(g.adminUpdateDataFile))
	mux.HandleFunc("POST /admin/api/preview", g.adminAPI(g.adminPreview))
	mux.HandleFunc("POST /admin/api/reload", g.adminAPI(g.adminReload))
}

func (g *gateway) adminAPI(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if !g.authorized(r) {
			g.metrics.authFailuresTotal.Add(1)
			writeError(w, http.StatusUnauthorized, "unauthorized")
			return
		}
		next(w, r)
	}
}

func (g *gateway) adminPage(w http.ResponseWriter, r *http.Request) {
	if !g.authorized(r) {
		g.metrics.authFailuresTotal.Add(1)
		writeError(w, http.StatusUnauthorized, "unauthorized")
		return
	}
	if key := r.URL.Query().Get("api_key"); key != "" && g.cfg.apiKey != "" {
		http.SetCookie(w, &http.Cookie{
			Name:     "kyc_gateway_api_key",
			Value:    key,
			Path:     "/admin",
			HttpOnly: true,
			SameSite: http.SameSiteLaxMode,
		})
	}
	tmpl, err := template.ParseFS(adminFS, "templates/admin.html")
	if err != nil {
		writeError(w, http.StatusInternalServerError, "admin template unavailable")
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	_ = tmpl.Execute(w, adminPageData{APIKeyConfigured: g.cfg.apiKey != "", DefaultAPIKey: ""})
}

func (g *gateway) adminStatic(w http.ResponseWriter, r *http.Request) {
	if !g.authorized(r) {
		g.metrics.authFailuresTotal.Add(1)
		writeError(w, http.StatusUnauthorized, "unauthorized")
		return
	}
	name := r.PathValue("name")
	if !safeNamePattern.MatchString(name) {
		writeError(w, http.StatusBadRequest, "invalid asset name")
		return
	}
	http.ServeFileFS(w, r, adminFS, "static/"+name)
}

func (g *gateway) adminConfig(w http.ResponseWriter, r *http.Request) {
	profiles, _ := g.readProfiles()
	writeJSON(w, http.StatusOK, adminConfigResponse{
		DocumentTypes: g.documentTypeSummaries(),
		DataFiles:     g.dataFileSummaries(),
		Profiles:      profiles,
	})
}

func (g *gateway) adminListDocumentTypes(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, g.documentTypeSummaries())
}

func (g *gateway) adminGetDocumentType(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	path, ok := g.documentTypePath(id)
	if !ok {
		writeError(w, http.StatusBadRequest, "invalid document type id")
		return
	}
	content, err := os.ReadFile(path)
	if err != nil {
		writeError(w, http.StatusNotFound, "document type not found")
		return
	}
	writeJSON(w, http.StatusOK, filePayload{Name: id, Content: string(content)})
}

func (g *gateway) adminCreateDocumentType(w http.ResponseWriter, r *http.Request) {
	var payload filePayload
	if !decodeJSON(w, r, &payload) {
		return
	}
	id, err := documentTypeID(payload.Name, payload.Content)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	path, ok := g.documentTypePath(id)
	if !ok {
		writeError(w, http.StatusBadRequest, "invalid document type id")
		return
	}
	if _, err := os.Stat(path); err == nil {
		writeError(w, http.StatusConflict, "document type already exists")
		return
	}
	if !g.validateDocumentType(w, r, payload.Content) {
		return
	}
	if err := g.atomicWrite(path, []byte(payload.Content), false); err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	g.writeReloadResult(w, r, http.StatusCreated, map[string]string{"status": "created", "id": id})
}

func (g *gateway) adminUpdateDocumentType(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	path, ok := g.documentTypePath(id)
	if !ok {
		writeError(w, http.StatusBadRequest, "invalid document type id")
		return
	}
	var payload filePayload
	if !decodeJSON(w, r, &payload) {
		return
	}
	if !g.validateDocumentType(w, r, payload.Content) {
		return
	}
	if err := g.atomicWrite(path, []byte(payload.Content), true); err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	g.writeReloadResult(w, r, http.StatusOK, map[string]string{"status": "saved", "id": id})
}

func (g *gateway) adminDuplicateDocumentType(w http.ResponseWriter, r *http.Request) {
	sourceID := r.PathValue("id")
	sourcePath, ok := g.documentTypePath(sourceID)
	if !ok {
		writeError(w, http.StatusBadRequest, "invalid source document type id")
		return
	}
	var payload duplicatePayload
	if !decodeJSON(w, r, &payload) {
		return
	}
	targetID := strings.TrimSpace(payload.ID)
	targetPath, ok := g.documentTypePath(targetID)
	if !ok {
		writeError(w, http.StatusBadRequest, "invalid target document type id")
		return
	}
	if _, err := os.Stat(targetPath); err == nil {
		writeError(w, http.StatusConflict, "target document type already exists")
		return
	}
	content, err := os.ReadFile(sourcePath)
	if err != nil {
		writeError(w, http.StatusNotFound, "source document type not found")
		return
	}
	updated := replaceDocumentTypeID(string(content), targetID)
	if !g.validateDocumentType(w, r, updated) {
		return
	}
	if err := g.atomicWrite(targetPath, []byte(updated), false); err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	g.writeReloadResult(w, r, http.StatusCreated, map[string]string{"status": "duplicated", "id": targetID})
}

func (g *gateway) adminDeleteDocumentType(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	path, ok := g.documentTypePath(id)
	if !ok {
		writeError(w, http.StatusBadRequest, "invalid document type id")
		return
	}
	if _, err := os.Stat(path); err != nil {
		writeError(w, http.StatusNotFound, "document type not found")
		return
	}
	if err := g.backupFile(path); err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	if err := os.Remove(path); err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	g.writeReloadResult(w, r, http.StatusOK, map[string]string{"status": "deleted", "id": id})
}

func (g *gateway) adminParseDocumentType(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(io.LimitReader(r.Body, 4*1024*1024))
	if err != nil {
		writeError(w, http.StatusBadRequest, "failed to read request body")
		return
	}
	g.proxyAdminBody(w, r, "/admin/parse/document-type", "application/x-yaml; charset=utf-8", body)
}

func (g *gateway) adminRenderDocumentType(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(io.LimitReader(r.Body, 4*1024*1024))
	if err != nil {
		writeError(w, http.StatusBadRequest, "failed to read request body")
		return
	}
	g.proxyAdminBody(w, r, "/admin/render/document-type", "application/json", body)
}

func (g *gateway) adminGetProfiles(w http.ResponseWriter, r *http.Request) {
	payload, err := g.readProfiles()
	if err != nil {
		writeError(w, http.StatusNotFound, "profiles config not found")
		return
	}
	writeJSON(w, http.StatusOK, payload)
}

func (g *gateway) adminUpdateProfiles(w http.ResponseWriter, r *http.Request) {
	var payload filePayload
	if !decodeJSON(w, r, &payload) {
		return
	}
	if !g.validateProfiles(w, r, payload.Content) {
		return
	}
	if err := g.atomicWrite(g.profilesPath(), []byte(payload.Content), true); err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	g.writeReloadResult(w, r, http.StatusOK, map[string]string{"status": "saved"})
}

func (g *gateway) adminListDataFiles(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, g.dataFileSummaries())
}

func (g *gateway) adminGetDataFile(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	path, ok := g.dataFilePath(name)
	if !ok {
		writeError(w, http.StatusBadRequest, "invalid data file name")
		return
	}
	content, err := os.ReadFile(path)
	if err != nil {
		writeError(w, http.StatusNotFound, "data file not found")
		return
	}
	writeJSON(w, http.StatusOK, filePayload{Name: name, Content: string(content)})
}

func (g *gateway) adminUpdateDataFile(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	path, ok := g.dataFilePath(name)
	if !ok {
		writeError(w, http.StatusBadRequest, "invalid data file name")
		return
	}
	var payload filePayload
	if !decodeJSON(w, r, &payload) {
		return
	}
	if err := g.atomicWrite(path, []byte(payload.Content), true); err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	g.writeReloadResult(w, r, http.StatusOK, map[string]string{"status": "saved", "name": name})
}

func (g *gateway) adminPreview(w http.ResponseWriter, r *http.Request) {
	if !isMultipart(r.Header.Get("Content-Type")) {
		writeError(w, http.StatusBadRequest, "multipart form upload required")
		return
	}
	body, contentType, query, err := previewMultipart(r, g.cfg.maxBodyBytes)
	if err != nil {
		if errors.Is(err, errBodyTooLarge) {
			writeError(w, http.StatusRequestEntityTooLarge, "request body too large")
			return
		}
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	base := g.upstreamURLWithQuery(query)
	req, err := http.NewRequestWithContext(r.Context(), http.MethodPost, base.String(), bytes.NewReader(body))
	if err != nil {
		writeError(w, http.StatusBadGateway, "failed to create upstream request")
		return
	}
	req.Header.Set("Content-Type", contentType)
	req.Header.Set("Accept", "application/json")
	req.ContentLength = int64(len(body))
	resp, err := g.client.Do(req)
	if err != nil {
		writeError(w, http.StatusBadGateway, "ocr upstream unavailable")
		return
	}
	defer resp.Body.Close()
	copyResponseHeaders(w.Header(), resp.Header)
	w.WriteHeader(resp.StatusCode)
	_, _ = io.Copy(w, resp.Body)
}

func (g *gateway) adminReload(w http.ResponseWriter, r *http.Request) {
	if err := g.reloadOCR(r.Context()); err != nil {
		writeError(w, http.StatusBadGateway, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "reloaded"})
}

func (g *gateway) writeReloadResult(w http.ResponseWriter, r *http.Request, status int, payload map[string]string) {
	if err := g.reloadOCR(r.Context()); err != nil {
		payload["reload_error"] = err.Error()
		writeJSON(w, http.StatusAccepted, payload)
		return
	}
	payload["reload"] = "ok"
	writeJSON(w, status, payload)
}

func (g *gateway) validateDocumentType(w http.ResponseWriter, r *http.Request, content string) bool {
	return g.validateUpstream(w, r, "/admin/validate/document-type", content)
}

func (g *gateway) validateProfiles(w http.ResponseWriter, r *http.Request, content string) bool {
	return g.validateUpstream(w, r, "/admin/validate/profiles", content)
}

func (g *gateway) validateUpstream(w http.ResponseWriter, r *http.Request, path, content string) bool {
	target := g.upstreamURLWithPath(path)
	req, err := http.NewRequestWithContext(r.Context(), http.MethodPost, target.String(), strings.NewReader(content))
	if err != nil {
		writeError(w, http.StatusBadGateway, "failed to create validation request")
		return false
	}
	req.Header.Set("Content-Type", "application/x-yaml; charset=utf-8")
	resp, err := g.client.Do(req)
	if err != nil {
		writeError(w, http.StatusBadGateway, "validation upstream unavailable")
		return false
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		return true
	}
	copyResponseHeaders(w.Header(), resp.Header)
	w.WriteHeader(resp.StatusCode)
	_, _ = io.Copy(w, resp.Body)
	return false
}

func (g *gateway) proxyAdminBody(w http.ResponseWriter, r *http.Request, path, contentType string, body []byte) {
	req, err := http.NewRequestWithContext(r.Context(), http.MethodPost, g.upstreamURLWithPath(path).String(), bytes.NewReader(body))
	if err != nil {
		writeError(w, http.StatusBadGateway, "failed to create upstream request")
		return
	}
	req.Header.Set("Content-Type", contentType)
	resp, err := g.client.Do(req)
	if err != nil {
		writeError(w, http.StatusBadGateway, "admin upstream unavailable")
		return
	}
	defer resp.Body.Close()
	copyResponseHeaders(w.Header(), resp.Header)
	w.WriteHeader(resp.StatusCode)
	_, _ = io.Copy(w, resp.Body)
}

func (g *gateway) reloadOCR(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, g.upstreamURLWithPath("/admin/reload").String(), nil)
	if err != nil {
		return err
	}
	resp, err := g.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return fmt.Errorf("ocr reload failed: %s", strings.TrimSpace(string(body)))
	}
	return nil
}

func (g *gateway) upstreamURLWithPath(path string) *url.URL {
	base := g.cfg.upstreamURLs[0]
	out := *base
	out.Path = joinPath(base.Path, path)
	out.RawQuery = ""
	return &out
}

func (g *gateway) upstreamURLWithQuery(values url.Values) *url.URL {
	index := (g.nextWorker.Add(1) - 1) % uint64(len(g.cfg.upstreamURLs))
	base := g.cfg.upstreamURLs[index]
	out := *base
	out.Path = joinPath(base.Path, "/ocr")
	out.RawQuery = withOCRDefaults(values, g.cfg.defaultAccuracy, g.cfg.defaultRetry).Encode()
	return &out
}

func (g *gateway) documentTypeSummaries() []documentTypeSummary {
	dir := filepath.Join(g.cfg.configDir, "document_types")
	entries, _ := os.ReadDir(dir)
	summaries := make([]documentTypeSummary, 0, len(entries))
	for _, entry := range entries {
		if entry.IsDir() || !isYAMLFile(entry.Name()) {
			continue
		}
		path := filepath.Join(dir, entry.Name())
		info, err := entry.Info()
		if err != nil {
			continue
		}
		id := strings.TrimSuffix(strings.TrimSuffix(entry.Name(), ".yaml"), ".yml")
		if content, err := os.ReadFile(path); err == nil {
			if parsed, err := documentTypeID(id, string(content)); err == nil {
				id = parsed
			}
		}
		summaries = append(summaries, documentTypeSummary{
			ID:       id,
			File:     filepath.ToSlash(filepath.Join("document_types", entry.Name())),
			Bytes:    info.Size(),
			Modified: info.ModTime().Format(time.RFC3339),
		})
	}
	sort.Slice(summaries, func(i, j int) bool { return summaries[i].ID < summaries[j].ID })
	return summaries
}

func (g *gateway) dataFileSummaries() []documentTypeSummary {
	entries, _ := os.ReadDir(g.cfg.dataDir)
	summaries := make([]documentTypeSummary, 0, len(entries))
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		if !isEditableDataFile(name) {
			continue
		}
		info, err := entry.Info()
		if err != nil {
			continue
		}
		summaries = append(summaries, documentTypeSummary{
			ID:       name,
			File:     name,
			Bytes:    info.Size(),
			Modified: info.ModTime().Format(time.RFC3339),
		})
	}
	sort.Slice(summaries, func(i, j int) bool { return summaries[i].ID < summaries[j].ID })
	return summaries
}

func (g *gateway) readProfiles() (filePayload, error) {
	content, err := os.ReadFile(g.profilesPath())
	if err != nil {
		return filePayload{}, err
	}
	return filePayload{Name: "document_profiles.yaml", Content: string(content)}, nil
}

func (g *gateway) profilesPath() string {
	return filepath.Join(g.cfg.configDir, "document_profiles.yaml")
}

func (g *gateway) documentTypePath(id string) (string, bool) {
	if !safeConfigName(id) {
		return "", false
	}
	path := filepath.Join(g.cfg.configDir, "document_types", id+".yaml")
	return path, true
}

func (g *gateway) dataFilePath(name string) (string, bool) {
	if !safeConfigName(name) || !isEditableDataFile(name) {
		return "", false
	}
	return filepath.Join(g.cfg.dataDir, name), true
}

func (g *gateway) atomicWrite(path string, content []byte, backup bool) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	if backup {
		if err := g.backupFile(path); err != nil && !errors.Is(err, os.ErrNotExist) {
			return err
		}
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".tmp-*")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)
	if _, err := tmp.Write(content); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmpName, path)
}

func (g *gateway) backupFile(path string) error {
	content, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	rel := filepath.Base(path)
	if configRel, err := filepath.Rel(g.cfg.configDir, path); err == nil && !strings.HasPrefix(configRel, "..") {
		rel = filepath.Join("config", configRel)
	} else if dataRel, err := filepath.Rel(g.cfg.dataDir, path); err == nil && !strings.HasPrefix(dataRel, "..") {
		rel = filepath.Join("data", dataRel)
	}
	target := filepath.Join(g.cfg.backupDir, time.Now().UTC().Format("20060102T150405Z"), rel)
	if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
		return err
	}
	return os.WriteFile(target, content, 0o644)
}

func previewMultipart(r *http.Request, limit int64) ([]byte, string, url.Values, error) {
	body, err := readLimited(r.Body, limit)
	if err != nil {
		return nil, "", nil, err
	}
	contentType := r.Header.Get("Content-Type")
	mediaType, params, err := multipartMediaType(contentType)
	if err != nil || !strings.EqualFold(mediaType, "multipart/form-data") {
		return nil, "", nil, errors.New("multipart form upload required")
	}
	reader := multipart.NewReader(bytes.NewReader(body), params["boundary"])
	form, err := reader.ReadForm(limit)
	if err != nil {
		return nil, "", nil, errors.New("failed to parse multipart upload")
	}
	defer form.RemoveAll()
	files := form.File["file"]
	if len(files) == 0 {
		return nil, "", nil, errors.New("file field is required")
	}
	var out bytes.Buffer
	writer := multipart.NewWriter(&out)
	for _, fileHeader := range files {
		src, err := fileHeader.Open()
		if err != nil {
			return nil, "", nil, err
		}
		header := make(textproto.MIMEHeader)
		header.Set("Content-Disposition", fmt.Sprintf(`form-data; name="file"; filename="%s"`, escapeQuotes(fileHeader.Filename)))
		if contentType := fileHeader.Header.Get("Content-Type"); contentType != "" {
			header.Set("Content-Type", contentType)
		}
		part, err := writer.CreatePart(header)
		if err != nil {
			_ = src.Close()
			return nil, "", nil, err
		}
		if _, err := io.Copy(part, src); err != nil {
			_ = src.Close()
			return nil, "", nil, err
		}
		_ = src.Close()
	}
	query := make(url.Values)
	for key, vals := range form.Value {
		if key == "" {
			continue
		}
		for _, value := range vals {
			query.Add(key, value)
		}
	}
	if err := writer.Close(); err != nil {
		return nil, "", nil, err
	}
	return out.Bytes(), writer.FormDataContentType(), query, nil
}

func escapeQuotes(value string) string {
	return strings.NewReplacer("\\", "\\\\", `"`, "\\\"").Replace(value)
}

func multipartMediaType(contentType string) (string, map[string]string, error) {
	mediaType, params, err := mime.ParseMediaType(contentType)
	return mediaType, params, err
}

func decodeJSON(w http.ResponseWriter, r *http.Request, target any) bool {
	defer r.Body.Close()
	decoder := json.NewDecoder(io.LimitReader(r.Body, 4*1024*1024))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		writeError(w, http.StatusBadRequest, "invalid json: "+err.Error())
		return false
	}
	return true
}

func documentTypeID(fallback, content string) (string, error) {
	for _, line := range strings.Split(content, "\n") {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "document_type:") || strings.HasPrefix(trimmed, "id:") {
			_, value, _ := strings.Cut(trimmed, ":")
			value = strings.Trim(strings.TrimSpace(value), `"'`)
			if value != "" {
				if !safeConfigName(value) {
					return "", errors.New("document_type contains unsupported characters")
				}
				return value, nil
			}
		}
	}
	fallback = strings.TrimSpace(fallback)
	if fallback == "" {
		return "", errors.New("document_type is required")
	}
	if !safeConfigName(fallback) {
		return "", errors.New("document_type contains unsupported characters")
	}
	return fallback, nil
}

func replaceDocumentTypeID(content, id string) string {
	lines := strings.Split(content, "\n")
	for i, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "document_type:") || strings.HasPrefix(trimmed, "id:") {
			prefix := line[:strings.Index(line, ":")+1]
			lines[i] = prefix + " " + id
			return strings.Join(lines, "\n")
		}
	}
	return "document_type: " + id + "\n" + content
}

func safeConfigName(name string) bool {
	name = strings.TrimSpace(name)
	return name != "" && safeNamePattern.MatchString(name) && !strings.Contains(name, "..")
}

func isYAMLFile(name string) bool {
	lower := strings.ToLower(name)
	return strings.HasSuffix(lower, ".yaml") || strings.HasSuffix(lower, ".yml")
}

func isEditableDataFile(name string) bool {
	lower := strings.ToLower(name)
	return strings.HasSuffix(lower, ".yaml") || strings.HasSuffix(lower, ".yml") || strings.HasSuffix(lower, ".txt")
}
