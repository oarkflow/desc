package main

import (
	"bytes"
	"context"
	"html/template"
	"io"
	"net/http"
	"strings"
)

type identityPageData struct {
	APIKeyConfigured bool
}

func (g *gateway) registerIdentityRoutes(mux *http.ServeMux) {
	mux.HandleFunc("GET /identity", g.identityPage)
	mux.HandleFunc("GET /identity/", g.identityPage)
	mux.HandleFunc("GET /identity/static/{name}", g.identityStatic)
	mux.HandleFunc("POST /identity/api/portrait", g.identityAPI("/identity/api/portrait"))
	mux.HandleFunc("POST /identity/api/liveness/frame", g.identityAPI("/identity/api/liveness/frame"))
	mux.HandleFunc("POST /identity/api/liveness/complete", g.identityAPI("/identity/api/liveness/complete"))
}

func (g *gateway) identityPage(w http.ResponseWriter, r *http.Request) {
	if !g.authorized(r) {
		g.metrics.authFailuresTotal.Add(1)
		writeError(w, http.StatusUnauthorized, "unauthorized")
		return
	}
	if key := r.URL.Query().Get("api_key"); key != "" && g.cfg.apiKey != "" {
		http.SetCookie(w, &http.Cookie{
			Name:     "kyc_gateway_api_key",
			Value:    key,
			Path:     "/",
			HttpOnly: true,
			SameSite: http.SameSiteLaxMode,
		})
	}
	tmpl, err := template.ParseFS(adminFS, "templates/identity.html")
	if err != nil {
		writeError(w, http.StatusInternalServerError, "identity template unavailable")
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	_ = tmpl.Execute(w, identityPageData{APIKeyConfigured: g.cfg.apiKey != ""})
}

func (g *gateway) identityStatic(w http.ResponseWriter, r *http.Request) {
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

func (g *gateway) identityAPI(path string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if !g.authorized(r) {
			g.metrics.authFailuresTotal.Add(1)
			writeError(w, http.StatusUnauthorized, "unauthorized")
			return
		}
		if path != "/identity/api/liveness/complete" && !isMultipart(r.Header.Get("Content-Type")) {
			writeError(w, http.StatusBadRequest, "multipart form upload required")
			return
		}
		body, err := readLimited(r.Body, g.cfg.maxBodyBytes)
		if err != nil {
			if err == errBodyTooLarge {
				g.metrics.requestTooLargeTotal.Add(1)
				writeError(w, http.StatusRequestEntityTooLarge, "request body too large")
				return
			}
			writeError(w, http.StatusBadRequest, "failed to read request body")
			return
		}

		upstream := g.upstreamURLWithPath(path)
		upstream.RawQuery = r.URL.RawQuery
		req, err := http.NewRequestWithContext(r.Context(), r.Method, upstream.String(), bytes.NewReader(body))
		if err != nil {
			writeError(w, http.StatusBadGateway, "failed to create identity upstream request")
			return
		}
		copyProxyHeaders(req.Header, r.Header)
		req.ContentLength = int64(len(body))

		resp, err := g.client.Do(req)
		if err != nil {
			status := http.StatusBadGateway
			if err == context.DeadlineExceeded || strings.Contains(err.Error(), "Client.Timeout") {
				status = http.StatusGatewayTimeout
			}
			g.metrics.upstreamErrorsTotal.Add(1)
			writeError(w, status, "identity upstream unavailable")
			return
		}
		defer resp.Body.Close()

		copyResponseHeaders(w.Header(), resp.Header)
		w.WriteHeader(resp.StatusCode)
		if _, err := io.Copy(w, resp.Body); err != nil {
			g.logger.Warn("copy upstream identity response failed", "error", err)
		}
	}
}
