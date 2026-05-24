const state = {
  apiKey: localStorage.getItem("kyc-admin-api-key") || "",
  types: [],
  dataFiles: [],
  currentType: "",
  currentData: "",
  documentModel: null,
  editorSection: "overview",
  selectedPass: null,
  selectedField: null,
  fieldQuery: "",
  previewJSON: null,
  regionImageFile: null,
  regionImageURL: "",
  regionImageLoaded: false,
  regionPreviewJSON: null,
  regions: [],
  selectedRegionId: null,
  regionDrag: null,
};

const $ = (id) => document.getElementById(id);
const STRATEGIES = ["same_line_after_label", "same_row_right_of_label", "same_row_left_of_label", "below_label", "append_following_line", "regex_from_full_text", "row_components", "anchor_region", "gazetteer_match", "template", "infer_unlabeled"];
const NORMALIZERS = ["nepali_digits_to_ascii", "citizenship_number", "bs_date_components", "ad_date_components", "clean_devanagari_name", "person_name_repair"];
const SOURCE_KINDS = ["printed", "handwritten"];
const PASS_MODES = ["default", "retry_only"];
const REGION_FIELD_TARGETS = new Set(["field_anchor", "field_retry", "tamper_field"]);
const REGION_COLORS = {
  field_anchor: "#0f766e",
  field_retry: "#2563eb",
  tamper_field: "#7c3aed",
  protected: "#b42318",
  expected_object: "#d97706",
  ocr: "#0891b2",
  object: "#65a30d",
  flag: "#dc2626",
};

function headers(extra = {}) {
  const out = { ...extra };
  if (state.apiKey) out["X-API-Key"] = state.apiKey;
  return out;
}

async function api(path, options = {}) {
  const res = await fetch(path, { ...options, headers: headers(options.headers || {}) });
  const text = await res.text();
  let payload = text;
  try { payload = text ? JSON.parse(text) : null; } catch {}
  if (!res.ok) {
    const message = payload && payload.error ? payload.error : text || res.statusText;
    throw new Error(message);
  }
  return payload;
}

function notice(message, isError = false) {
  const box = $("notice");
  box.textContent = message;
  box.hidden = false;
  box.classList.toggle("error", isError);
  clearTimeout(notice.timer);
  notice.timer = setTimeout(() => { box.hidden = true; }, isError ? 9000 : 4500);
}

function switchView(name) {
  document.querySelectorAll(".tab").forEach((tab) => tab.classList.toggle("is-active", tab.dataset.view === name));
  document.querySelectorAll(".view").forEach((view) => view.classList.toggle("is-active", view.id === name));
}

async function loadConfig() {
  const config = await api("/admin/api/config");
  state.types = config.document_types || [];
  state.dataFiles = config.data_files || [];
  $("profilesYaml").value = config.profiles ? config.profiles.content : "";
  renderTypeList();
  renderDataList();
  renderPreviewTypes();
}

function renderTypeList() {
  const list = $("typeList");
  list.innerHTML = "";
  state.types.forEach((item) => {
    const button = document.createElement("button");
    button.className = item.id === state.currentType ? "is-active" : "";
    button.innerHTML = `<strong>${escapeHTML(item.id)}</strong><small>${escapeHTML(item.file)} · ${escapeHTML(item.modified || "")}</small>`;
    button.addEventListener("click", () => loadDocumentType(item.id));
    list.appendChild(button);
  });
}

function renderDataList() {
  const list = $("dataList");
  list.innerHTML = "";
  state.dataFiles.forEach((item) => {
    const button = document.createElement("button");
    button.className = item.id === state.currentData ? "is-active" : "";
    button.innerHTML = `<strong>${escapeHTML(item.id)}</strong><small>${item.bytes} bytes · ${escapeHTML(item.modified || "")}</small>`;
    button.addEventListener("click", () => loadDataFile(item.id));
    list.appendChild(button);
  });
}

function renderPreviewTypes() {
  const select = $("previewDocumentType");
  select.innerHTML = `<option value="">Auto detect</option>`;
  state.types.forEach((item) => {
    const opt = document.createElement("option");
    opt.value = item.id;
    opt.textContent = item.id;
    select.appendChild(opt);
  });
}

async function loadDocumentType(id) {
  const payload = await api(`/admin/api/document-types/${encodeURIComponent(id)}`);
  const parsed = await api("/admin/api/parse/document-type", {
    method: "POST",
    headers: { "Content-Type": "application/x-yaml" },
    body: payload.content,
  });
  state.currentType = id;
  state.documentModel = normalizeDocumentModel(parsed);
  state.editorSection = "overview";
  state.selectedPass = null;
  state.selectedField = null;
  state.fieldQuery = "";
  hydrateRegionsFromModel();
  $("typeTitle").textContent = id;
  $("typeMeta").textContent = payload.name || "";
  $("saveType").disabled = false;
  $("duplicateType").disabled = false;
  $("deleteType").disabled = false;
  renderTypeList();
  renderDocumentForm();
  await refreshGeneratedYAML();
}

function newDocumentModel(id = "new_document_type") {
  return {
    document_type: id,
    profile: {
      detect: { cues: [], min_score: 1 },
      ocr: { timeout_seconds: null, max_passes: null, max_image_side: null, retry_padding_px: 24, passes: [defaultPass()] },
      fields: {},
    },
  };
}

function normalizeDocumentModel(parsed) {
  const model = {
    document_type: parsed.document_type || state.currentType || "document_type",
    profile: parsed.profile || {},
  };
  model.profile.detect ||= { cues: [], min_score: 1 };
  model.profile.detect.cues ||= [];
  model.profile.ocr ||= {};
  model.profile.ocr.passes ||= [defaultPass()];
  model.profile.fields ||= {};
  return model;
}

function defaultPass() {
  return {
    name: "default",
    mode: "default",
    lang: "ne",
    script: "",
    source_kind: "printed",
    text_detection_model: "",
    text_recognition_model: "",
    ocr_version: "",
    min_confidence: null,
    retry_fields: [],
    run_if_missing_fields: [],
    run_if_below_confidence: null,
    timeout_seconds: null,
    preprocessing: { upscale: true, denoise: false, threshold: false, crop_border: true, enhance: true, clean_background: false, max_image_side: null },
  };
}

function defaultField() {
  return {
    labels: [],
    strategies: [],
    validators: [],
    normalizers: [],
    components: [],
    regex: "",
    gazetteer: "",
    review_threshold: null,
    anchor_region: null,
    source_passes: [],
    source_kinds: [],
      review_source_kinds: [],
      retry_region: null,
      retry_regions: [],
      anchor_regions: [],
      same_as_field: "",
    consistency_field: "",
    gazetteer_hint_fields: [],
    template: "",
  };
}

function renderDocumentForm() {
  const model = state.documentModel;
  const root = $("documentForm");
  if (!model) {
    root.innerHTML = `<div class="empty-state">Select or create a document type.</div>`;
    return;
  }
  root.innerHTML = "";
  root.appendChild(documentStats(model));

  const workbench = document.createElement("div");
  workbench.className = "config-workbench";
  workbench.appendChild(configNavigator(model));
  workbench.appendChild(configDetail(model));
  root.appendChild(workbench);
}

function documentStats(model) {
  const profile = model.profile || {};
  const passes = profile.ocr?.passes || [];
  const fields = Object.keys(profile.fields || {});
  const cues = profile.detect?.cues || [];
  const el = document.createElement("div");
  el.className = "doc-stats";
  el.innerHTML = `
    <span><strong>${escapeHTML(model.document_type || "Untitled")}</strong></span>
    <span>${passes.length} OCR passes</span>
    <span>${fields.length} fields</span>
    <span>${cues.length} cues</span>
  `;
  return el;
}

function configNavigator(model) {
  const nav = document.createElement("aside");
  nav.className = "config-nav";
  nav.appendChild(navGroup("Setup", [
    navItem("Overview", state.editorSection === "overview", () => selectEditor("overview")),
    navItem("Detection", state.editorSection === "detection", () => selectEditor("detection")),
    navItem("OCR runtime", state.editorSection === "runtime", () => selectEditor("runtime")),
    navItem("Regions", state.editorSection === "regions", () => selectEditor("regions"), `${state.regions.length} boxes`),
  ]));

  const passes = model.profile.ocr?.passes || [];
  nav.appendChild(navGroup("OCR passes", [
    ...passes.map((pass, index) => navItem(pass.name || `Pass ${index + 1}`, state.editorSection === "pass" && state.selectedPass === index, () => selectPass(index), passMeta(pass))),
    navAction("Add pass", () => {
      model.profile.ocr.passes ||= [];
      model.profile.ocr.passes.push(defaultPass());
      selectPass(model.profile.ocr.passes.length - 1);
    }),
  ]));

  const fields = model.profile.fields || {};
  const fieldNames = Object.keys(fields).filter((name) => fieldMatchesQuery(name, fields[name]));
  const fieldSearch = document.createElement("input");
  fieldSearch.className = "nav-search";
  fieldSearch.placeholder = "Filter fields";
  fieldSearch.value = state.fieldQuery;
  fieldSearch.addEventListener("input", () => {
    state.fieldQuery = fieldSearch.value.trim().toLowerCase();
    renderDocumentForm();
  });
  const fieldGroupItems = [fieldSearch];
  fieldGroupItems.push(navItem("All fields", state.editorSection === "fields", () => selectEditor("fields"), `${Object.keys(fields).length} configured`));
  fieldNames.forEach((name) => {
    fieldGroupItems.push(navItem(name, state.editorSection === "field" && state.selectedField === name, () => selectField(name), fieldSummary(fields[name])));
  });
  fieldGroupItems.push(navAction("Add field", () => {
    const name = uniqueName("new_field", Object.keys(fields));
    fields[name] = defaultField();
    selectField(name);
  }));
  nav.appendChild(navGroup("Fields", fieldGroupItems));
  return nav;
}

function configDetail(model) {
  const detail = document.createElement("section");
  detail.className = "config-detail";
  if (state.editorSection === "overview") {
    detail.appendChild(section("Document overview", [
      inputControl("Document type", model.document_type, (value) => { model.document_type = slugValue(value); updateTitle(); }),
      numberControl("Detection min score", model.profile.detect.min_score, (value) => { model.profile.detect.min_score = value; }, 1, 0),
      numberControl("OCR timeout seconds", model.profile.ocr.timeout_seconds, (value) => { model.profile.ocr.timeout_seconds = value; }, 1, 0),
      numberControl("Max passes", model.profile.ocr.max_passes, (value) => { model.profile.ocr.max_passes = value; }, 1, 0),
      numberControl("Max image side", model.profile.ocr.max_image_side, (value) => { model.profile.ocr.max_image_side = value; }, 1, 0),
      numberControl("Retry padding px", model.profile.ocr.retry_padding_px ?? 24, (value) => { model.profile.ocr.retry_padding_px = value ?? 24; }, 1, 0),
    ]));
    detail.appendChild(summaryStrip(model));
    return detail;
  }
  if (state.editorSection === "detection") {
    detail.appendChild(detectSection(model.profile.detect));
    return detail;
  }
  if (state.editorSection === "runtime") {
    detail.appendChild(runtimeSection(model.profile.ocr));
    return detail;
  }
  if (state.editorSection === "regions") {
    detail.appendChild(regionEditorSection(model));
    return detail;
  }
  if (state.editorSection === "pass") {
    const passes = model.profile.ocr?.passes || [];
    const index = state.selectedPass ?? 0;
    if (passes[index]) detail.appendChild(passCard(passes[index], index, passes, true));
    return detail;
  }
  if (state.editorSection === "field") {
    const fields = model.profile.fields || {};
    const name = state.selectedField;
    if (name && fields[name]) detail.appendChild(fieldCard(fields, name, fields[name], Object.keys(fields).indexOf(name), true));
    return detail;
  }
  detail.appendChild(fieldsSection(model.profile.fields || {}));
  return detail;
}

function selectEditor(sectionName) {
  state.editorSection = sectionName;
  state.selectedPass = null;
  state.selectedField = null;
  renderDocumentForm();
}

function selectPass(index) {
  state.editorSection = "pass";
  state.selectedPass = index;
  state.selectedField = null;
  renderDocumentForm();
}

function selectField(name) {
  state.editorSection = "field";
  state.selectedField = name;
  state.selectedPass = null;
  renderDocumentForm();
}

function navGroup(title, items) {
  const group = document.createElement("div");
  group.className = "nav-group";
  group.innerHTML = `<div class="nav-title">${escapeHTML(title)}</div>`;
  items.forEach((item) => group.appendChild(item));
  return group;
}

function navItem(label, active, onClick, meta = "") {
  const item = document.createElement("button");
  item.type = "button";
  item.className = active ? "nav-item is-active" : "nav-item";
  item.innerHTML = `<span>${escapeHTML(label)}</span>${meta ? `<small>${escapeHTML(meta)}</small>` : ""}`;
  item.addEventListener("click", onClick);
  return item;
}

function navAction(label, onClick) {
  const item = document.createElement("button");
  item.type = "button";
  item.className = "nav-action";
  item.textContent = label;
  item.addEventListener("click", onClick);
  return item;
}

function identitySection(model) {
  return section("Identity", [
    inputControl("Document type", model.document_type, (value) => { model.document_type = slugValue(value); state.currentType ||= model.document_type; updateTitle(); }),
  ]);
}

function detectSection(detect) {
  return section("Detection", [
    numberControl("Minimum cue score", detect.min_score, (value) => { detect.min_score = value; }, 1, 0),
    listEditor("Cues", detect.cues.map((cue) => cue.text || ""), (values) => { detect.cues = values.map((text) => ({ text })); }, "Cue text"),
  ]);
}

function ocrSection(ocr) {
  const wrapper = runtimeSection(ocr);
  const passes = repeatEditor("OCR passes", ocr.passes || [], () => defaultPass(), (pass, index) => passCard(pass, index, ocr.passes, true), () => renderDocumentForm());
  wrapper.appendChild(passes);
  return wrapper;
}

function runtimeSection(ocr) {
  return section("OCR runtime", [
    numberControl("Timeout seconds", ocr.timeout_seconds, (value) => { ocr.timeout_seconds = value; }, 1, 0),
    numberControl("Max passes", ocr.max_passes, (value) => { ocr.max_passes = value; }, 1, 0),
    numberControl("Max image side", ocr.max_image_side, (value) => { ocr.max_image_side = value; }, 1, 0),
    numberControl("Retry padding px", ocr.retry_padding_px ?? 24, (value) => { ocr.retry_padding_px = value ?? 24; }, 1, 0),
  ]);
}

function passCard(pass, index, passes, open = true) {
  pass.preprocessing ||= defaultPass().preprocessing;
  return card(`Pass ${index + 1}: ${pass.name || "unnamed"}`, [
    inputControl("Name", pass.name, (value) => { pass.name = slugValue(value); }),
    selectControl("Mode", pass.mode || "default", PASS_MODES, (value) => { pass.mode = value; }),
    inputControl("Language", pass.lang || "ne", (value) => { pass.lang = value; }),
    inputControl("Script", pass.script || "", (value) => { pass.script = value; }),
    selectControl("Source kind", pass.source_kind || "printed", SOURCE_KINDS, (value) => { pass.source_kind = value; }),
    inputControl("Detection model", pass.text_detection_model || "", (value) => { pass.text_detection_model = value; }),
    inputControl("Recognition model", pass.text_recognition_model || "", (value) => { pass.text_recognition_model = value; }),
    inputControl("OCR version", pass.ocr_version || "", (value) => { pass.ocr_version = value; }),
    decimalControl("Min confidence", pass.min_confidence, (value) => { pass.min_confidence = value; }),
    decimalControl("Run if below confidence", pass.run_if_below_confidence, (value) => { pass.run_if_below_confidence = value; }),
    numberControl("Timeout seconds", pass.timeout_seconds, (value) => { pass.timeout_seconds = value; }, 1, 0),
    tagControl("Retry fields", pass.retry_fields || [], (values) => { pass.retry_fields = values; }, fieldNames()),
    tagControl("Run if missing fields", pass.run_if_missing_fields || [], (values) => { pass.run_if_missing_fields = values; }, fieldNames()),
    preprocessingEditor(pass.preprocessing),
    removeButton("Remove pass", () => { passes.splice(index, 1); renderDocumentForm(); }),
  ], open, passMeta(pass));
}

function preprocessingEditor(pre) {
  return subsection("Preprocessing", [
    boolControl("Upscale", pre.upscale, (value) => { pre.upscale = value; }),
    boolControl("Denoise", pre.denoise, (value) => { pre.denoise = value; }),
    boolControl("Threshold", pre.threshold, (value) => { pre.threshold = value; }),
    boolControl("Crop border", pre.crop_border, (value) => { pre.crop_border = value; }),
    boolControl("Enhance", pre.enhance, (value) => { pre.enhance = value; }),
    boolControl("Clean background", pre.clean_background, (value) => { pre.clean_background = value; }),
    numberControl("Max image side", pre.max_image_side, (value) => { pre.max_image_side = value; }, 1, 0),
  ]);
}

function fieldsSection(fields) {
  const wrapper = section("Fields", []);
  const toolbar = document.createElement("div");
  toolbar.className = "section-toolbar";
  const filter = document.createElement("input");
  filter.className = "toolbar-search";
  filter.placeholder = "Filter fields";
  filter.value = state.fieldQuery;
  filter.addEventListener("input", () => {
    state.fieldQuery = filter.value.trim().toLowerCase();
    renderDocumentForm();
  });
  const add = button("Add field", "button primary", () => {
    const name = uniqueName("new_field", Object.keys(fields));
    fields[name] = defaultField();
    selectField(name);
  });
  const expand = button("Expand all", "button", () => setCardsOpen(wrapper, true));
  const collapse = button("Collapse all", "button", () => setCardsOpen(wrapper, false));
  toolbar.append(filter, expand, collapse, add);
  wrapper.appendChild(toolbar);
  const entries = Object.entries(fields).filter(([name, field]) => fieldMatchesQuery(name, field));
  if (!entries.length) {
    const empty = document.createElement("div");
    empty.className = "empty-state compact";
    empty.textContent = Object.keys(fields).length ? "No fields match the filter." : "No fields configured yet.";
    wrapper.appendChild(empty);
    return wrapper;
  }
  entries.forEach(([name, field], index) => wrapper.appendChild(fieldCard(fields, name, field, index, false)));
  return wrapper;
}

function fieldCard(fields, name, field, index, open = false) {
  const fieldConfig = { ...defaultField(), ...field };
  Object.assign(field, fieldConfig);
  return card(`Field ${index + 1}: ${name}`, [
    inputControl("Field name", name, (value) => {
      const next = slugValue(value);
      if (!next || next === name || fields[next]) return;
      fields[next] = fields[name];
      delete fields[name];
      renderDocumentForm();
    }),
    listEditor("Labels", field.labels || [], (values) => { field.labels = values; }, "Label text"),
    multiSelectControl("Strategies", field.strategies || [], STRATEGIES, (values) => { field.strategies = values; }),
    inputControl("Regex", field.regex || "", (value) => { field.regex = value; }),
    validatorEditor(field.validators || [], (values) => { field.validators = values; }),
    multiSelectControl("Normalizers", field.normalizers || [], NORMALIZERS, (values) => { field.normalizers = values; }),
    tagControl("Components", field.components || [], (values) => { field.components = values; }, ["Year", "Month", "Day"]),
    inputControl("Template", field.template || "", (value) => { field.template = value; }),
    inputControl("Gazetteer path", field.gazetteer || "", (value) => { field.gazetteer = value; }),
    decimalControl("Review threshold", field.review_threshold, (value) => { field.review_threshold = value; }),
    regionControl("Anchor region", field.anchor_region, (value) => { field.anchor_region = value; }),
    regionControl("Retry region", field.retry_region, (value) => { field.retry_region = value; }),
    tagControl("Source passes", field.source_passes || [], (values) => { field.source_passes = values; }, passNames()),
    multiSelectControl("Source kinds", field.source_kinds || [], SOURCE_KINDS, (values) => { field.source_kinds = values; }),
    multiSelectControl("Review source kinds", field.review_source_kinds || [], SOURCE_KINDS, (values) => { field.review_source_kinds = values; }),
    selectControl("Same as field", field.same_as_field || "", ["", ...fieldNames()], (value) => { field.same_as_field = value; }),
    selectControl("Consistency field", field.consistency_field || "", ["", ...fieldNames()], (value) => { field.consistency_field = value; }),
    tagControl("Gazetteer hint fields", field.gazetteer_hint_fields || [], (values) => { field.gazetteer_hint_fields = values; }, fieldNames()),
    removeButton("Remove field", () => { delete fields[name]; renderDocumentForm(); }),
  ], open, fieldSummary(field));
}

function section(title, children) {
  const el = document.createElement("section");
  el.className = "form-section";
  el.innerHTML = `<div class="section-title"><h3>${escapeHTML(title)}</h3></div>`;
  const grid = document.createElement("div");
  grid.className = "control-grid";
  children.forEach((child) => grid.appendChild(child));
  el.appendChild(grid);
  return el;
}

function subsection(title, children) {
  const el = document.createElement("div");
  el.className = "subsection";
  el.innerHTML = `<h4>${escapeHTML(title)}</h4>`;
  const grid = document.createElement("div");
  grid.className = "control-grid compact";
  children.forEach((child) => grid.appendChild(child));
  el.appendChild(grid);
  return el;
}

function card(title, children, open = true, meta = "") {
  const el = document.createElement("details");
  el.className = "form-card";
  el.open = open;
  el.innerHTML = `<summary><span>${escapeHTML(title)}</span>${meta ? `<small>${escapeHTML(meta)}</small>` : ""}</summary>`;
  const grid = document.createElement("div");
  grid.className = "control-grid";
  children.forEach((child) => grid.appendChild(child));
  el.appendChild(grid);
  return el;
}

function repeatEditor(title, items, createItem, renderItem, rerender) {
  const el = document.createElement("div");
  el.className = "repeat-block";
  const head = document.createElement("div");
  head.className = "repeat-head";
  head.innerHTML = `<h3>${escapeHTML(title)}</h3>`;
  head.appendChild(button("Add", "button", () => { items.push(createItem()); rerender(); }));
  el.appendChild(head);
  items.forEach((item, index) => el.appendChild(renderItem(item, index)));
  return el;
}

function inputControl(label, value, onChange) {
  const el = controlShell(label);
  const input = document.createElement("input");
  input.value = value ?? "";
  input.addEventListener("input", () => { onChange(input.value); scheduleYAMLRefresh(); });
  el.appendChild(input);
  return el;
}

function numberControl(label, value, onChange, step = 1, min = null) {
  const el = controlShell(label);
  const input = document.createElement("input");
  input.type = "number";
  input.step = String(step);
  if (min !== null) input.min = String(min);
  input.value = value ?? "";
  input.addEventListener("input", () => { onChange(input.value === "" ? null : Number(input.value)); scheduleYAMLRefresh(); });
  el.appendChild(input);
  return el;
}

function decimalControl(label, value, onChange) {
  return numberControl(label, value, onChange, "0.01", 0);
}

function boolControl(label, value, onChange) {
  const el = document.createElement("label");
  el.className = "toggle-control";
  const input = document.createElement("input");
  input.type = "checkbox";
  input.checked = Boolean(value);
  input.addEventListener("change", () => { onChange(input.checked); scheduleYAMLRefresh(); });
  el.append(input, document.createTextNode(label));
  return el;
}

function selectControl(label, value, options, onChange) {
  const el = controlShell(label);
  const select = document.createElement("select");
  options.forEach((option) => {
    const opt = document.createElement("option");
    opt.value = option;
    opt.textContent = option || "None";
    select.appendChild(opt);
  });
  select.value = value ?? "";
  select.addEventListener("change", () => { onChange(select.value); scheduleYAMLRefresh(); });
  el.appendChild(select);
  return el;
}

function multiSelectControl(label, values, options, onChange) {
  const el = controlShell(label);
  const box = document.createElement("div");
  box.className = "checkbox-grid";
  options.forEach((option) => {
    const item = document.createElement("label");
    item.className = "mini-check";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = values.includes(option);
    input.addEventListener("change", () => {
      const next = new Set(values);
      input.checked ? next.add(option) : next.delete(option);
      values.splice(0, values.length, ...next);
      onChange(values);
      scheduleYAMLRefresh();
    });
    item.append(input, document.createTextNode(option));
    box.appendChild(item);
  });
  el.appendChild(box);
  return el;
}

function tagControl(label, values, onChange, suggestions = []) {
  const el = controlShell(label);
  const input = document.createElement("input");
  input.value = values.join(", ");
  input.setAttribute("list", `${label.replace(/\W+/g, "-")}-list`);
  input.addEventListener("input", () => {
    const next = splitCSV(input.value);
    values.splice(0, values.length, ...next);
    onChange(values);
    scheduleYAMLRefresh();
  });
  el.appendChild(input);
  if (suggestions.length) {
    const datalist = document.createElement("datalist");
    datalist.id = input.getAttribute("list");
    suggestions.forEach((suggestion) => {
      const option = document.createElement("option");
      option.value = suggestion;
      datalist.appendChild(option);
    });
    el.appendChild(datalist);
  }
  return el;
}

function listEditor(label, values, onChange, placeholder) {
  const el = document.createElement("div");
  el.className = "list-editor";
  el.innerHTML = `<div class="inline-head"><span>${escapeHTML(label)}</span></div>`;
  const rows = document.createElement("div");
  rows.className = "list-editor-rows";
  values.forEach((value, index) => rows.appendChild(listRow(value, placeholder, (next) => { values[index] = next; onChange(values.filter(Boolean)); scheduleYAMLRefresh(); }, () => { values.splice(index, 1); onChange(values); renderDocumentForm(); })));
  el.querySelector(".inline-head").appendChild(button("Add", "button", () => { values.push(""); onChange(values); renderDocumentForm(); }));
  el.appendChild(rows);
  return el;
}

function listRow(value, placeholder, onChange, onRemove) {
  const row = document.createElement("div");
  row.className = "inline-row";
  const input = document.createElement("input");
  input.placeholder = placeholder;
  input.value = value;
  input.addEventListener("input", () => onChange(input.value));
  row.appendChild(input);
  row.appendChild(removeButton("Remove", onRemove));
  return row;
}

function validatorEditor(values, onChange) {
  const el = document.createElement("div");
  el.className = "list-editor";
  el.innerHTML = `<div class="inline-head"><span>Validators</span></div>`;
  const rows = document.createElement("div");
  rows.className = "list-editor-rows";
  values.forEach((validator, index) => {
    const row = document.createElement("div");
    row.className = "inline-row two";
    const type = document.createElement("input");
    type.placeholder = "type";
    type.value = validator.type || "regex";
    const pattern = document.createElement("input");
    pattern.placeholder = "pattern";
    pattern.value = validator.pattern || "";
    [type, pattern].forEach((input) => input.addEventListener("input", () => {
      values[index] = { type: type.value, pattern: pattern.value };
      onChange(values);
      scheduleYAMLRefresh();
    }));
    row.append(type, pattern, removeButton("Remove", () => { values.splice(index, 1); onChange(values); renderDocumentForm(); }));
    rows.appendChild(row);
  });
  el.querySelector(".inline-head").appendChild(button("Add", "button", () => { values.push({ type: "regex", pattern: "" }); onChange(values); renderDocumentForm(); }));
  el.appendChild(rows);
  return el;
}

function regionControl(label, value, onChange) {
  const region = Array.isArray(value) ? value : ["", "", "", ""];
  const el = controlShell(label);
  const row = document.createElement("div");
  row.className = "region-row";
  ["x1", "y1", "x2", "y2"].forEach((name, index) => {
    const input = document.createElement("input");
    input.type = "number";
    input.step = "0.001";
    input.min = "0";
    input.max = "1";
    input.placeholder = name;
    input.value = region[index] ?? "";
    input.addEventListener("input", () => {
      region[index] = input.value === "" ? "" : Number(input.value);
      const next = region.every((item) => item !== "" && Number.isFinite(Number(item))) ? region.map(Number) : null;
      onChange(next);
      scheduleYAMLRefresh();
    });
    row.appendChild(input);
  });
  el.appendChild(row);
  return el;
}

function controlShell(label) {
  const el = document.createElement("label");
  el.className = "control";
  const span = document.createElement("span");
  span.textContent = label;
  el.appendChild(span);
  return el;
}

function button(text, className, onClick) {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = className;
  btn.textContent = text;
  btn.addEventListener("click", onClick);
  return btn;
}

function removeButton(text, onClick) {
  return button(text, "button danger compact-button", onClick);
}

function updateTitle() {
  $("typeTitle").textContent = state.documentModel?.document_type || "Document type";
}

async function refreshGeneratedYAML() {
  if (!state.documentModel) return;
  const cleaned = cleanDocumentModel(state.documentModel);
  const rendered = await api("/admin/api/render/document-type", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(cleaned),
  });
  $("typeYaml").value = rendered.content || "";
}

function scheduleYAMLRefresh() {
  updateTitle();
  clearTimeout(scheduleYAMLRefresh.timer);
  scheduleYAMLRefresh.timer = setTimeout(() => refreshGeneratedYAML().catch(() => {}), 350);
}

function cleanDocumentModel(model) {
  return pruneEmpty(JSON.parse(JSON.stringify(model)));
}

function pruneEmpty(value) {
  if (Array.isArray(value)) return value.map(pruneEmpty).filter((item) => item !== "" && item !== null && item !== undefined);
  if (value && typeof value === "object") {
    Object.keys(value).forEach((key) => {
      value[key] = pruneEmpty(value[key]);
      if (value[key] === "" || value[key] === null || value[key] === undefined) delete value[key];
      if (Array.isArray(value[key]) && value[key].length === 0) delete value[key];
      if (value[key] && typeof value[key] === "object" && !Array.isArray(value[key]) && Object.keys(value[key]).length === 0) delete value[key];
    });
  }
  return value;
}

async function loadDataFile(name) {
  const payload = await api(`/admin/api/data/${encodeURIComponent(name)}`);
  state.currentData = name;
  $("dataContent").value = payload.content;
  $("dataTitle").textContent = name;
  $("dataMeta").textContent = `${payload.content.length} characters`;
  $("saveData").disabled = false;
  renderDataList();
}

async function saveDocumentType() {
  if (!state.documentModel) return;
  await refreshGeneratedYAML();
  const id = state.documentModel.document_type.trim();
  const body = JSON.stringify({ name: id, content: $("typeYaml").value });
  const path = state.currentType ? `/admin/api/document-types/${encodeURIComponent(state.currentType)}` : "/admin/api/document-types";
  const method = state.currentType ? "PUT" : "POST";
  const payload = await api(path, { method, headers: { "Content-Type": "application/json" }, body });
  state.currentType = payload.id || id;
  notice(payload.reload_error ? `Saved, but reload failed: ${payload.reload_error}` : "Document type saved and OCR reload requested.", Boolean(payload.reload_error));
  await loadConfig();
  await loadDocumentType(state.currentType);
}

async function duplicateDocumentType() {
  if (!state.currentType) return;
  const target = prompt("New document type id", `${state.currentType}_copy`);
  if (!target) return;
  await api(`/admin/api/document-types/${encodeURIComponent(state.currentType)}/duplicate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id: target.trim() }),
  });
  notice("Document type duplicated.");
  await loadConfig();
  await loadDocumentType(target.trim());
}

async function deleteDocumentType() {
  if (!state.currentType || !confirm(`Delete ${state.currentType}? A backup will be kept.`)) return;
  await api(`/admin/api/document-types/${encodeURIComponent(state.currentType)}`, { method: "DELETE" });
  state.currentType = "";
  state.documentModel = null;
  $("typeYaml").value = "";
  $("typeTitle").textContent = "Select a document type";
  $("documentForm").innerHTML = "";
  $("saveType").disabled = $("duplicateType").disabled = $("deleteType").disabled = true;
  notice("Document type moved to backup and OCR reload requested.");
  await loadConfig();
}

async function saveProfiles() {
  await api("/admin/api/profiles", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name: "document_profiles.yaml", content: $("profilesYaml").value }),
  });
  notice("Global OCR settings saved and OCR reload requested.");
}

async function saveData() {
  if (!state.currentData) return;
  await api(`/admin/api/data/${encodeURIComponent(state.currentData)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name: state.currentData, content: $("dataContent").value }),
  });
  notice("Data file saved and OCR reload requested.");
  await loadConfig();
}

async function runPreview() {
  const form = $("previewForm");
  const data = new FormData();
  const file = $("previewFile").files[0];
  if (!file) {
    notice("Choose a document image first.", true);
    return;
  }
  data.append("file", file, file.name);
  Array.from(form.elements).forEach((el) => {
    if (!el.name || el.name === "file") return;
    if (el.type === "checkbox") {
      data.append(el.name, el.checked ? el.value : "false");
      return;
    }
    data.append(el.name, el.value);
  });
  $("previewResult").textContent = "Running OCR...";
  const res = await fetch("/admin/api/preview", { method: "POST", headers: headers(), body: data });
  const text = await res.text();
  if (!res.ok) throw new Error(text || res.statusText);
  try {
    state.previewJSON = JSON.parse(text);
    $("previewResult").textContent = JSON.stringify(state.previewJSON, null, 2);
  } catch {
    state.previewJSON = null;
    $("previewResult").textContent = text;
  }
  loadPreviewImage(file);
}

function loadPreviewImage(file) {
  const image = $("previewImage");
  image.onload = drawOverlay;
  image.src = URL.createObjectURL(file);
}

function drawOverlay() {
  const image = $("previewImage");
  const canvas = $("previewCanvas");
  const rect = image.getBoundingClientRect();
  canvas.width = rect.width;
  canvas.height = rect.height;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const result = state.previewJSON || {};
  const items = Array.isArray(result.items) ? result.items : [];
  ctx.strokeStyle = "#0f766e";
  ctx.lineWidth = 2;
  items.forEach((item) => {
    if (!Array.isArray(item.box) || item.box.length < 2) return;
    const xs = item.box.map((p) => Number(p[0])).filter(Number.isFinite);
    const ys = item.box.map((p) => Number(p[1])).filter(Number.isFinite);
    if (!xs.length || !ys.length || !result.width || !result.height) return;
    const x = Math.min(...xs) / result.width * canvas.width;
    const y = Math.min(...ys) / result.height * canvas.height;
    const w = (Math.max(...xs) - Math.min(...xs)) / result.width * canvas.width;
    const h = (Math.max(...ys) - Math.min(...ys)) / result.height * canvas.height;
    ctx.strokeRect(x, y, w, h);
  });
}

function fieldNames() {
  return Object.keys(state.documentModel?.profile?.fields || {});
}

function passNames() {
  return (state.documentModel?.profile?.ocr?.passes || []).map((item) => item.name).filter(Boolean);
}

function passMeta(pass) {
  return [pass.mode || "default", pass.lang || "ne", pass.source_kind || "printed"].filter(Boolean).join(" · ");
}

function fieldSummary(field) {
  const parts = [];
  if (field.strategies?.length) parts.push(`${field.strategies.length} strategies`);
  if (field.labels?.length) parts.push(`${field.labels.length} labels`);
  if (field.source_passes?.length) parts.push(field.source_passes.join(", "));
  if (field.anchor_region) parts.push("anchor");
  if (field.retry_region) parts.push("retry");
  if (field.review_threshold !== null && field.review_threshold !== undefined) parts.push(`review ${field.review_threshold}`);
  return parts.join(" · ") || "No extraction rules";
}

function fieldMatchesQuery(name, field) {
  if (!state.fieldQuery) return true;
  const haystack = [
    name,
    ...(field.labels || []),
    ...(field.strategies || []),
    ...(field.normalizers || []),
    ...(field.source_passes || []),
    field.regex || "",
    field.template || "",
  ].join(" ").toLowerCase();
  return haystack.includes(state.fieldQuery);
}

function setCardsOpen(root, open) {
  root.querySelectorAll(".form-card").forEach((card) => {
    card.open = open;
  });
}

function summaryStrip(model) {
  const fields = model.profile.fields || {};
  const passes = model.profile.ocr?.passes || [];
  const el = document.createElement("div");
  el.className = "overview-grid";
  el.innerHTML = `
    <div><strong>Detection</strong><span>${(model.profile.detect?.cues || []).length} cues, score ${model.profile.detect?.min_score ?? 1}</span></div>
    <div><strong>OCR passes</strong><span>${passes.map((pass) => pass.name).filter(Boolean).join(", ") || "default"}</span></div>
    <div><strong>Fields</strong><span>${Object.keys(fields).join(", ") || "No fields configured"}</span></div>
  `;
  return el;
}

function regionEditorSection() {
  const el = document.createElement("section");
  el.className = "form-section region-editor";
  el.innerHTML = `
    <div class="section-title">
      <h3>Regions</h3>
      <div class="actions">
        <button id="regionRunOCR" class="button" type="button">Run OCR preview</button>
        <button id="regionDelete" class="button danger" type="button" disabled>Delete selected</button>
      </div>
    </div>
    <div class="region-workbench">
      <div class="region-canvas-panel">
        <label class="region-upload">Image<input id="regionImageFile" type="file" accept="image/png,image/jpeg,image/webp"></label>
        <div id="regionCanvasWrap" class="region-canvas-wrap">
          <div class="drop-hint">Drop image here, then drag to create a box</div>
          <img id="regionImage" alt="">
          <canvas id="regionCanvas"></canvas>
        </div>
      </div>
      <aside class="region-side">
        <div class="region-block">
          <h2>Selected box</h2>
          <label>Target
            <select id="regionTarget">
              <option value="field_anchor">Field anchor region</option>
              <option value="field_retry">Field retry region</option>
              <option value="tamper_field">Tamper field/layout region</option>
              <option value="protected">Protected asset region</option>
              <option value="expected_object">Expected object region</option>
            </select>
          </label>
          <label id="regionFieldWrap">Field
            <input id="regionFieldName" list="regionFieldNames" placeholder="full_name">
          </label>
          <datalist id="regionFieldNames"></datalist>
          <label id="regionAssetWrap">Asset or object
            <input id="regionAssetName" list="regionAssetNames" placeholder="photo">
          </label>
          <datalist id="regionAssetNames">
            <option value="photo"></option><option value="portrait"></option><option value="face"></option><option value="logo"></option>
            <option value="hologram"></option><option value="stamp"></option><option value="signature"></option><option value="seal"></option>
          </datalist>
          <div class="region-coords">
            <label>x1<input id="regionX1" type="number" min="0" max="1" step="0.0001"></label>
            <label>y1<input id="regionY1" type="number" min="0" max="1" step="0.0001"></label>
            <label>x2<input id="regionX2" type="number" min="0" max="1" step="0.0001"></label>
            <label>y2<input id="regionY2" type="number" min="0" max="1" step="0.0001"></label>
          </div>
        </div>
        <div class="region-block">
          <div class="inline-head"><h2>Boxes</h2><button id="regionClear" class="button danger" type="button">Clear</button></div>
          <div id="regionList" class="region-list"></div>
        </div>
        <div class="region-block">
          <h2>Selected crop</h2>
          <div class="region-crop-wrap"><canvas id="regionCropCanvas"></canvas><div id="regionCropEmpty" class="crop-empty">Select a box to inspect its crop.</div></div>
        </div>
        <div class="region-block">
          <h2>OCR preview</h2>
          <div id="regionPreviewSummary" class="preview-summary muted">Run OCR preview to draw OCR boxes on the image.</div>
          <details class="raw-json-details">
            <summary>Raw JSON</summary>
            <pre id="regionPreviewResult" class="result-box compact"></pre>
          </details>
        </div>
      </aside>
    </div>
  `;
  setTimeout(wireRegionEditor, 0);
  return el;
}

function hydrateRegionsFromModel() {
  state.regions = regionsFromModel(state.documentModel);
  state.selectedRegionId = state.regions[0]?.id || null;
  state.regionPreviewJSON = null;
}

function regionsFromModel(model) {
  if (!model) return [];
  const out = [];
  const fields = model.profile?.fields || {};
  Object.entries(fields).forEach(([name, field]) => {
    regionBoxes(field.anchor_region).forEach((box) => out.push(regionItem("field_anchor", name, box)));
    regionBoxes(field.anchor_regions).forEach((box) => out.push(regionItem("field_anchor", name, box)));
    regionBoxes(field.retry_region).forEach((box) => out.push(regionItem("field_retry", name, box)));
    regionBoxes(field.retry_regions).forEach((box) => out.push(regionItem("field_retry", name, box)));
  });
  Object.entries(model.profile?.tamper?.field_regions || {}).forEach(([name, value]) => {
    regionBoxes(value).forEach((box) => out.push(regionItem("tamper_field", name, box)));
  });
  Object.entries(model.profile?.tamper?.protected_regions || {}).forEach(([name, value]) => {
    regionBoxes(value).forEach((box) => out.push(regionItem("protected", name, box)));
  });
  (model.profile?.tamper?.expected_objects || []).forEach((item) => {
    regionBoxes(item.region).forEach((box) => out.push(regionItem("expected_object", item.label || "face", box)));
    regionBoxes(item.regions).forEach((box) => out.push(regionItem("expected_object", item.label || "face", box)));
  });
  return out;
}

function regionItem(target, name, box) {
  return { id: newRegionId(), target, name, box: cleanRegionBox(box) };
}

function wireRegionEditor() {
  if (!$("regionCanvas")) return;
  renderRegionInspector();
  restoreRegionImage();
  $("regionImageFile").addEventListener("change", (event) => {
    const file = event.target.files[0];
    if (file) loadRegionImage(file);
  });
  const wrap = $("regionCanvasWrap");
  wrap.addEventListener("dragover", (event) => { event.preventDefault(); wrap.classList.add("is-dragover"); });
  wrap.addEventListener("dragleave", () => wrap.classList.remove("is-dragover"));
  wrap.addEventListener("drop", (event) => {
    event.preventDefault();
    wrap.classList.remove("is-dragover");
    const file = event.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) loadRegionImage(file);
  });
  $("regionRunOCR").addEventListener("click", () => runRegionPreview().catch((err) => notice(err.message, true)));
  $("regionDelete").addEventListener("click", () => {
    state.regions = state.regions.filter((item) => item.id !== state.selectedRegionId);
    state.selectedRegionId = state.regions[0]?.id || null;
    regionChanged();
  });
  $("regionClear").addEventListener("click", () => {
    if (!confirm("Clear all drawn regions for this document type?")) return;
    state.regions = [];
    state.selectedRegionId = null;
    regionChanged();
  });
  ["regionTarget", "regionFieldName", "regionAssetName", "regionX1", "regionY1", "regionX2", "regionY2"].forEach((id) => {
    $(id).addEventListener("input", updateSelectedRegionFromForm);
    $(id).addEventListener("change", updateSelectedRegionFromForm);
  });
  const canvas = $("regionCanvas");
  canvas.addEventListener("pointerdown", onRegionPointerDown);
  canvas.addEventListener("pointermove", onRegionPointerMove);
  window.addEventListener("pointerup", onRegionPointerUp);
  drawRegions();
}

function restoreRegionImage() {
  if (!state.regionImageURL || !$("regionImage")) return;
  const image = $("regionImage");
  image.onload = () => {
    state.regionImageLoaded = true;
    $("regionCanvasWrap").classList.add("has-image");
    drawRegions();
  };
  image.src = state.regionImageURL;
}

function loadRegionImage(file) {
  state.regionImageFile = file;
  state.regionPreviewJSON = null;
  if (state.regionImageURL) URL.revokeObjectURL(state.regionImageURL);
  state.regionImageURL = URL.createObjectURL(file);
  restoreRegionImage();
}

function selectedRegion() {
  return state.regions.find((item) => item.id === state.selectedRegionId) || null;
}

function renderRegionInspector() {
  if (!$("regionList")) return;
  const selected = selectedRegion();
  $("regionDelete").disabled = !selected;
  ["regionTarget", "regionFieldName", "regionAssetName", "regionX1", "regionY1", "regionX2", "regionY2"].forEach((id) => { $(id).disabled = !selected; });
  const fieldList = $("regionFieldNames");
  fieldList.innerHTML = "";
  fieldNames().forEach((name) => {
    const option = document.createElement("option");
    option.value = name;
    fieldList.appendChild(option);
  });
  if (selected) {
    $("regionTarget").value = selected.target;
    $("regionFieldName").value = REGION_FIELD_TARGETS.has(selected.target) ? selected.name : "";
    $("regionAssetName").value = REGION_FIELD_TARGETS.has(selected.target) ? "" : selected.name;
    selected.box.forEach((value, index) => { $(["regionX1", "regionY1", "regionX2", "regionY2"][index]).value = value; });
  } else {
    ["regionFieldName", "regionAssetName", "regionX1", "regionY1", "regionX2", "regionY2"].forEach((id) => { $(id).value = ""; });
  }
  $("regionFieldWrap").hidden = !selected || !REGION_FIELD_TARGETS.has($("regionTarget").value);
  $("regionAssetWrap").hidden = !selected || REGION_FIELD_TARGETS.has($("regionTarget").value);
  renderRegionList();
  drawSelectedRegionCrop();
}

function renderRegionList() {
  const list = $("regionList");
  list.innerHTML = "";
  if (!state.regions.length) {
    const empty = document.createElement("div");
    empty.className = "region-item";
    empty.innerHTML = "<strong>No boxes yet</strong><small>Drag on the image to create one.</small>";
    list.appendChild(empty);
    return;
  }
  state.regions.forEach((item, index) => {
    const row = document.createElement("button");
    row.type = "button";
    row.className = item.id === state.selectedRegionId ? "region-item is-active" : "region-item";
    row.innerHTML = `<strong>${index + 1}. ${escapeHTML(regionLabel(item))}</strong><small>${item.box.join(", ")}</small>`;
    row.addEventListener("click", () => { state.selectedRegionId = item.id; renderRegionInspector(); drawRegions(); });
    list.appendChild(row);
  });
}

function regionLabel(item) {
  return `${item.target.replaceAll("_", " ")}: ${item.name || "unnamed"}`;
}

function updateSelectedRegionFromForm() {
  const item = selectedRegion();
  if (!item) return;
  item.target = $("regionTarget").value;
  item.name = REGION_FIELD_TARGETS.has(item.target)
    ? slugValue($("regionFieldName").value)
    : slugValue($("regionAssetName").value);
  item.box = cleanRegionBox(["regionX1", "regionY1", "regionX2", "regionY2"].map((id) => Number($(id).value)));
  regionChanged();
}

function regionChanged() {
  syncRegionsToModel();
  renderRegionInspector();
  drawRegions();
  scheduleYAMLRefresh();
}

function syncRegionsToModel() {
  if (!state.documentModel) return;
  const profile = state.documentModel.profile ||= {};
  profile.fields ||= {};
  profile.tamper ||= {};
  const existingExpectedObjects = profile.tamper.expected_objects || [];
  profile.tamper.field_regions = {};
  profile.tamper.protected_regions = {};
  Object.values(profile.fields).forEach((field) => {
    delete field.anchor_region;
    delete field.anchor_regions;
    delete field.retry_region;
    delete field.retry_regions;
  });
  const grouped = {};
  state.regions.forEach((item) => {
    if (!item.name) return;
    grouped[item.target] ||= {};
    grouped[item.target][item.name] ||= [];
    grouped[item.target][item.name].push(cleanRegionBox(item.box));
  });
  Object.entries(grouped.field_anchor || {}).forEach(([name, boxes]) => {
    profile.fields[name] ||= defaultField();
    profile.fields[name].strategies ||= [];
    if (!profile.fields[name].strategies.includes("anchor_region")) profile.fields[name].strategies.push("anchor_region");
    assignRegionBoxes(profile.fields[name], "anchor_region", "anchor_regions", boxes);
  });
  Object.entries(grouped.field_retry || {}).forEach(([name, boxes]) => {
    profile.fields[name] ||= defaultField();
    assignRegionBoxes(profile.fields[name], "retry_region", "retry_regions", boxes);
  });
  Object.entries(grouped.tamper_field || {}).forEach(([name, boxes]) => {
    profile.tamper.field_regions[name] = serializeRegionBoxes(boxes);
  });
  Object.entries(grouped.protected || {}).forEach(([name, boxes]) => {
    profile.tamper.protected_regions[name] = serializeRegionBoxes(boxes);
  });
  const expectedByLabel = new Map(existingExpectedObjects.map((item) => [item.label, { ...item }]));
  Object.entries(grouped.expected_object || {}).forEach(([label, boxes]) => {
    const existing = expectedByLabel.get(label) || { label, required: false, min_confidence: 0 };
    const next = { ...existing, label };
    delete next.region;
    delete next.regions;
    assignRegionBoxes(next, "region", "regions", boxes);
    expectedByLabel.set(label, next);
  });
  profile.tamper.expected_objects = Array.from(expectedByLabel.values());
}

function assignRegionBoxes(target, singleKey, multiKey, boxes) {
  delete target[singleKey];
  delete target[multiKey];
  if (boxes.length === 1) target[singleKey] = cleanRegionBox(boxes[0]);
  if (boxes.length > 1) target[multiKey] = boxes.map(cleanRegionBox);
}

function serializeRegionBoxes(boxes) {
  return boxes.length === 1 ? cleanRegionBox(boxes[0]) : boxes.map(cleanRegionBox);
}

async function runRegionPreview() {
  if (!state.regionImageFile) throw new Error("Upload an image in Regions first.");
  const data = new FormData();
  data.append("file", state.regionImageFile, state.regionImageFile.name);
  if (state.currentType) data.append("document_type", state.currentType);
  data.append("values_only", "false");
  data.append("accuracy_mode", "accurate");
  data.append("retry", "true");
  $("regionPreviewSummary").textContent = "Running OCR preview...";
  $("regionPreviewResult").textContent = "";
  const res = await fetch("/admin/api/preview", { method: "POST", headers: headers(), body: data });
  const text = await res.text();
  if (!res.ok) throw new Error(text || res.statusText);
  state.regionPreviewJSON = JSON.parse(text);
  renderRegionPreviewSummary();
  $("regionPreviewResult").textContent = JSON.stringify(state.regionPreviewJSON, null, 2);
  drawRegions();
  drawSelectedRegionCrop();
}

function renderRegionPreviewSummary() {
  const box = $("regionPreviewSummary");
  if (!box) return;
  const result = state.regionPreviewJSON || {};
  const values = result.values || {};
  const flags = result.flags || [];
  const objects = result.object_summary || {};
  const faceNote = objects.face_count
    ? `${objects.face_count} heuristic face/photo candidate${objects.face_count === 1 ? "" : "s"}`
    : "no face/photo candidates";
  const fieldText = Object.keys(values).length
    ? Object.entries(values).map(([key, value]) => `<span><strong>${escapeHTML(key)}</strong>${escapeHTML(value)}</span>`).join("")
    : "<span>No fields extracted</span>";
  box.className = "preview-summary";
  box.innerHTML = `
    <div><strong>${escapeHTML(result.document_type || "unknown")}</strong><small>Document type</small></div>
    <div><strong>${Number(objects.text_region_count || 0)}</strong><small>OCR text boxes drawn</small></div>
    <div><strong>${escapeHTML(faceNote)}</strong><small>OpenCV heuristic, not identity proof</small></div>
    <div><strong>${flags.length}</strong><small>Tamper flags drawn when box evidence exists</small></div>
    <div class="preview-fields">${fieldText}</div>
  `;
}

function drawRegions() {
  const canvas = $("regionCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  if (!state.regionImageLoaded) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    return;
  }
  const metrics = regionImageMetrics();
  ctx.clearRect(0, 0, metrics.width, metrics.height);
  drawRegionPreview(ctx, metrics);
  state.regions.forEach((item) => drawRegionBox(ctx, metrics, item, item.id === state.selectedRegionId));
  drawSelectedRegionCrop();
}

function drawSelectedRegionCrop() {
  const canvas = $("regionCropCanvas");
  const empty = $("regionCropEmpty");
  const image = $("regionImage");
  const item = selectedRegion();
  if (!canvas || !empty) return;
  if (!state.regionImageLoaded || !item || !image.naturalWidth || !image.naturalHeight) {
    canvas.width = 0;
    canvas.height = 0;
    empty.hidden = false;
    return;
  }
  const [x1, y1, x2, y2] = item.box;
  const sx = Math.round(x1 * image.naturalWidth);
  const sy = Math.round(y1 * image.naturalHeight);
  const sw = Math.max(Math.round((x2 - x1) * image.naturalWidth), 1);
  const sh = Math.max(Math.round((y2 - y1) * image.naturalHeight), 1);
  const maxWidth = 320;
  const scale = Math.max(1, Math.min(4, maxWidth / Math.max(sw, 1)));
  canvas.width = Math.round(sw * scale);
  canvas.height = Math.round(sh * scale);
  const ctx = canvas.getContext("2d");
  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(image, sx, sy, sw, sh, 0, 0, canvas.width, canvas.height);
  empty.hidden = true;
}

function regionImageMetrics() {
  const image = $("regionImage");
  const canvas = $("regionCanvas");
  const rect = image.getBoundingClientRect();
  canvas.width = Math.round(rect.width);
  canvas.height = Math.round(rect.height);
  canvas.style.width = `${rect.width}px`;
  canvas.style.height = `${rect.height}px`;
  canvas.style.left = `${image.offsetLeft}px`;
  canvas.style.top = `${image.offsetTop}px`;
  return { width: canvas.width, height: canvas.height };
}

function drawRegionBox(ctx, metrics, item, active) {
  const [x1, y1, x2, y2] = item.box;
  const x = x1 * metrics.width;
  const y = y1 * metrics.height;
  const w = (x2 - x1) * metrics.width;
  const h = (y2 - y1) * metrics.height;
  ctx.save();
  ctx.strokeStyle = REGION_COLORS[item.target] || "#0f766e";
  ctx.fillStyle = `${ctx.strokeStyle}22`;
  ctx.lineWidth = active ? 3 : 2;
  ctx.fillRect(x, y, w, h);
  ctx.strokeRect(x, y, w, h);
  ctx.fillStyle = ctx.strokeStyle;
  ctx.font = "12px system-ui";
  ctx.fillText(regionLabel(item), x + 5, Math.max(y + 15, 15));
  if (active) [[x, y], [x + w, y], [x, y + h], [x + w, y + h]].forEach(([px, py]) => {
    ctx.fillStyle = "#fff";
    ctx.strokeStyle = "#111827";
    ctx.fillRect(px - 6, py - 6, 12, 12);
    ctx.strokeRect(px - 6, py - 6, 12, 12);
  });
  ctx.restore();
}

function drawRegionPreview(ctx, metrics) {
  const result = state.regionPreviewJSON || {};
  const sourceWidth = Number(result.width) || $("regionImage").naturalWidth;
  const sourceHeight = Number(result.height) || $("regionImage").naturalHeight;
  if (!sourceWidth || !sourceHeight) return;
  (result.items || []).forEach((item) => {
    const bounds = boundsFromOCRBox(item.box);
    if (bounds) drawPixelRegion(ctx, metrics, bounds, sourceWidth, sourceHeight, REGION_COLORS.ocr, item.text || "ocr");
  });
  (result.objects || []).forEach((item) => {
    const pixel = item.box?.pixel;
    if (pixel) drawPixelRegion(ctx, metrics, [pixel.x, pixel.y, pixel.x + pixel.width, pixel.y + pixel.height], sourceWidth, sourceHeight, REGION_COLORS.object, item.label || "object");
  });
  (result.flags || []).forEach((flag) => {
    const bounds = flag.evidence?.bounds;
    if (isRegionBox(bounds)) drawPixelRegion(ctx, metrics, bounds, sourceWidth, sourceHeight, REGION_COLORS.flag, flag.code || "flag");
  });
}

function drawPixelRegion(ctx, metrics, bounds, sourceWidth, sourceHeight, color, label) {
  const [x1, y1, x2, y2] = bounds.map(Number);
  const x = x1 / sourceWidth * metrics.width;
  const y = y1 / sourceHeight * metrics.height;
  const w = (x2 - x1) / sourceWidth * metrics.width;
  const h = (y2 - y1) / sourceHeight * metrics.height;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.setLineDash([5, 4]);
  ctx.strokeRect(x, y, w, h);
  ctx.fillStyle = color;
  ctx.font = "11px system-ui";
  ctx.fillText(label, x + 4, Math.max(y + 12, 12));
  ctx.restore();
}

function onRegionPointerDown(event) {
  if (!state.regionImageLoaded) return;
  event.preventDefault();
  $("regionCanvas").setPointerCapture(event.pointerId);
  const point = regionPointer(event);
  const hit = hitRegion(point);
  if (hit) {
    state.selectedRegionId = hit.id;
    const corner = hitRegionCorner(hit, point);
    state.regionDrag = { type: corner ? "resize" : "move", corner, pointerId: event.pointerId, start: point, original: [...hit.box] };
  } else {
    const target = $("regionTarget")?.value || "field_anchor";
    const item = regionItem(target, defaultRegionName(target), [point.x, point.y, point.x, point.y]);
    state.regions.push(item);
    state.selectedRegionId = item.id;
    state.regionDrag = { type: "create", pointerId: event.pointerId, start: point, original: [...item.box] };
  }
  regionChanged();
}

function onRegionPointerMove(event) {
  if (!state.regionImageLoaded) return;
  updateRegionCursor(event);
  if (!state.regionDrag) return;
  event.preventDefault();
  const item = selectedRegion();
  if (!item) return;
  const point = regionPointer(event);
  const drag = state.regionDrag;
  if (drag.type === "move") {
    const dx = point.x - drag.start.x;
    const dy = point.y - drag.start.y;
    const width = drag.original[2] - drag.original[0];
    const height = drag.original[3] - drag.original[1];
    const x1 = clamp(drag.original[0] + dx, 0, 1 - width);
    const y1 = clamp(drag.original[1] + dy, 0, 1 - height);
    item.box = cleanRegionBox([x1, y1, x1 + width, y1 + height]);
  } else if (drag.type === "resize") {
    const next = [...drag.original];
    if (drag.corner.includes("w")) next[0] = point.x;
    if (drag.corner.includes("e")) next[2] = point.x;
    if (drag.corner.includes("n")) next[1] = point.y;
    if (drag.corner.includes("s")) next[3] = point.y;
    item.box = cleanRegionBox(next);
  } else {
    item.box = cleanRegionBox([drag.start.x, drag.start.y, point.x, point.y]);
  }
  regionChanged();
}

function onRegionPointerUp() {
  if (state.regionDrag?.pointerId) {
    try { $("regionCanvas").releasePointerCapture(state.regionDrag.pointerId); } catch {}
  }
  state.regionDrag = null;
  const item = selectedRegion();
  if (item && (item.box[2] - item.box[0] < 0.004 || item.box[3] - item.box[1] < 0.004)) {
    state.regions = state.regions.filter((candidate) => candidate.id !== item.id);
    state.selectedRegionId = state.regions[0]?.id || null;
    regionChanged();
  }
  drawSelectedRegionCrop();
}

function regionPointer(event) {
  const rect = $("regionCanvas").getBoundingClientRect();
  return { x: clamp((event.clientX - rect.left) / rect.width), y: clamp((event.clientY - rect.top) / rect.height) };
}

function hitRegion(point) {
  for (let i = state.regions.length - 1; i >= 0; i -= 1) {
    const item = state.regions[i];
    const [x1, y1, x2, y2] = item.box;
    if (point.x >= x1 && point.x <= x2 && point.y >= y1 && point.y <= y2) return item;
  }
  return null;
}

function hitRegionCorner(item, point) {
  const threshold = 0.025;
  const corners = [["nw", item.box[0], item.box[1]], ["ne", item.box[2], item.box[1]], ["sw", item.box[0], item.box[3]], ["se", item.box[2], item.box[3]]];
  const found = corners.find(([, x, y]) => Math.abs(point.x - x) <= threshold && Math.abs(point.y - y) <= threshold);
  return found ? found[0] : "";
}

function updateRegionCursor(event) {
  if (state.regionDrag || !$("regionCanvas")) return;
  const hit = hitRegion(regionPointer(event));
  if (!hit) {
    $("regionCanvas").style.cursor = "crosshair";
    return;
  }
  const corner = hitRegionCorner(hit, regionPointer(event));
  $("regionCanvas").style.cursor = corner === "nw" || corner === "se" ? "nwse-resize" : corner ? "nesw-resize" : "move";
}

function regionBoxes(value) {
  if (!value) return [];
  if (isRegionBox(value)) return [cleanRegionBox(value)];
  if (!Array.isArray(value)) return [];
  return value.filter(isRegionBox).map(cleanRegionBox);
}

function isRegionBox(value) {
  return Array.isArray(value) && value.length === 4 && value.every((item) => Number.isFinite(Number(item)));
}

function cleanRegionBox(value) {
  const [rawX1, rawY1, rawX2, rawY2] = value.map(Number);
  const x1 = clamp(Math.min(rawX1, rawX2));
  const y1 = clamp(Math.min(rawY1, rawY2));
  const x2 = clamp(Math.max(rawX1, rawX2));
  const y2 = clamp(Math.max(rawY1, rawY2));
  return [round4(x1), round4(y1), round4(x2), round4(y2)];
}

function boundsFromOCRBox(box) {
  if (!Array.isArray(box) || box.length < 2) return null;
  const xs = box.map((point) => Number(point[0])).filter(Number.isFinite);
  const ys = box.map((point) => Number(point[1])).filter(Number.isFinite);
  if (!xs.length || !ys.length) return null;
  return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
}

function defaultRegionName(target) {
  if (REGION_FIELD_TARGETS.has(target)) return fieldNames()[0] || "";
  return target === "expected_object" ? "face" : "photo";
}

function clamp(value, min = 0, max = 1) {
  return Math.min(Math.max(Number(value) || 0, min), max);
}

function round4(value) {
  return Math.round(Number(value) * 10000) / 10000;
}

function newRegionId() {
  if (window.crypto && typeof window.crypto.randomUUID === "function") return window.crypto.randomUUID();
  return `region-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function splitCSV(value) {
  return value.split(",").map((item) => item.trim()).filter(Boolean);
}

function slugValue(value) {
  return value.trim().replace(/\s+/g, "_");
}

function uniqueName(base, existing) {
  let name = base;
  let i = 2;
  while (existing.includes(name)) name = `${base}_${i++}`;
  return name;
}

function escapeHTML(value) {
  return String(value).replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
}

function wireEvents() {
  $("apiKey").value = state.apiKey;
  $("apiKey").addEventListener("input", (event) => {
    state.apiKey = event.target.value;
    localStorage.setItem("kyc-admin-api-key", state.apiKey);
  });
  document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => switchView(tab.dataset.view)));
  $("newType").addEventListener("click", async () => {
    state.currentType = "";
    state.documentModel = newDocumentModel();
    state.editorSection = "overview";
    state.selectedPass = null;
    state.selectedField = null;
    state.fieldQuery = "";
    hydrateRegionsFromModel();
    $("typeTitle").textContent = "New document type";
    $("saveType").disabled = false;
    $("duplicateType").disabled = true;
    $("deleteType").disabled = true;
    renderDocumentForm();
    await refreshGeneratedYAML();
  });
  $("saveType").addEventListener("click", () => saveDocumentType().catch((err) => notice(err.message, true)));
  $("duplicateType").addEventListener("click", () => duplicateDocumentType().catch((err) => notice(err.message, true)));
  $("deleteType").addEventListener("click", () => deleteDocumentType().catch((err) => notice(err.message, true)));
  $("saveProfiles").addEventListener("click", () => saveProfiles().catch((err) => notice(err.message, true)));
  $("saveData").addEventListener("click", () => saveData().catch((err) => notice(err.message, true)));
  $("runPreview").addEventListener("click", () => runPreview().catch((err) => notice(err.message, true)));
  window.addEventListener("resize", drawOverlay);
}

wireEvents();
loadConfig().catch((err) => notice(err.message, true));
