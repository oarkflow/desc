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
};

const $ = (id) => document.getElementById(id);
const STRATEGIES = ["same_line_after_label", "same_row_right_of_label", "same_row_left_of_label", "below_label", "append_following_line", "regex_from_full_text", "row_components", "anchor_region", "gazetteer_match", "template", "infer_unlabeled"];
const NORMALIZERS = ["nepali_digits_to_ascii", "citizenship_number", "bs_date_components", "ad_date_components", "clean_devanagari_name", "person_name_repair"];
const SOURCE_KINDS = ["printed", "handwritten"];
const PASS_MODES = ["default", "retry_only"];

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
