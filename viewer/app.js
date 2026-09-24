"use strict";

/* Phylodigital Tree Viewer — UI layer.
 *
 * Rendering and event wiring only. Every derived quantity (tree layout,
 * matrix, labels, colors) comes from viewer/layout.js so the logic stays
 * testable under `node --test`.
 */

(function () {
  const L = window.PhylodigyLayout;

  const els = {
    app: document.getElementById("app"),
    badge: document.getElementById("artifact-badge"),
    detail: document.getElementById("side-detail"),
    diagnostics: document.getElementById("diagnostics-list"),
    digest: document.getElementById("digest-line"),
    drop: document.getElementById("drop-zone"),
    evidenceBoundary: document.getElementById("evidence-boundary"),
    evidenceEmpty: document.getElementById("evidence-empty"),
    evidenceTable: document.getElementById("evidence-table"),
    fileInput: document.getElementById("file-input"),
    genomes: document.getElementById("genome-table"),
    matrix: document.getElementById("distance-matrix"),
    raw: document.getElementById("raw-json"),
    sampleButton: document.getElementById("sample-button"),
    scale: document.getElementById("tree-scale"),
    svg: document.getElementById("tree-svg"),
    treeCanvas: document.getElementById("tree-canvas"),
    violationList: document.getElementById("violation-list"),
    violationPanel: document.getElementById("violation-panel"),
  };

  const NS = "http://www.w3.org/2000/svg";
  let state = { lineage: null, names: new Map(), selected: null, layoutMode: "rect" };

  /* ---------- data intake ---------- */

  function acceptText(text, sourceLabel) {
    let parsed;
    try {
      parsed = JSON.parse(text);
    } catch (error) {
      showError(`Not valid JSON (${sourceLabel}): ${error.message}`);
      return;
    }
    const tree = parsed && parsed.tree;
    if (!tree || !Array.isArray(tree.edges) || !Array.isArray(tree.taxa)) {
      showError(
        `This JSON is not a phylodigy.architecture_lineage artifact ` +
        `(missing tree.edges / tree.taxa). Got artifact_type: ` +
        `${(parsed && parsed.artifact_type) || "unknown"}.`
      );
      return;
    }
    state = { lineage: parsed, names: L.genomeNames(parsed), selected: null };
    renderAll();
  }

  function showError(message) {
    const node = document.createElement("p");
    node.className = "drop-sub";
    node.style.color = "var(--bad, #b91c1c)";
    node.textContent = message;
    els.drop.querySelector(".drop-inner").appendChild(node);
    els.drop.hidden = false;
  }

  els.fileInput.addEventListener("change", (event) => {
    const file = event.target.files && event.target.files[0];
    if (!file) return;
    file.text().then((text) => acceptText(text, file.name));
    event.target.value = "";
  });

  for (const name of ["dragenter", "dragover"]) {
    els.drop.addEventListener(name, (event) => {
      event.preventDefault();
      els.drop.classList.add("dragging");
    });
  }
  els.drop.addEventListener("dragleave", () => els.drop.classList.remove("dragging"));
  els.drop.addEventListener("drop", (event) => {
    event.preventDefault();
    els.drop.classList.remove("dragging");
    const file = event.dataTransfer.files && event.dataTransfer.files[0];
    if (file) file.text().then((text) => acceptText(text, file.name));
  });

  els.sampleButton.addEventListener("click", () => loadSample(true));

  async function loadSample(forceFetch) {
    // Bundled regenerated artifact: `phylodigy toy-tree -o viewer/toy-lineage.json`.
    // Embedded fallback keeps the page functional over file:// where fetch is blocked.
    if (!forceFetch && window.EMBEDDED_SAMPLE) return acceptText(window.EMBEDDED_SAMPLE, "embedded sample");
    try {
      const response = await fetch("toy-lineage.json");
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      acceptText(await response.text(), "toy-lineage.json");
    } catch (error) {
      if (window.EMBEDDED_SAMPLE) {
        acceptText(window.EMBEDDED_SAMPLE, "embedded sample");
      } else {
        showError(`Could not load the bundled sample: ${error.message}. Open a lineage JSON instead.`);
      }
    }
  }

  /* ---------- tabs ---------- */

  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("is-active", t === tab));
      document.querySelectorAll(".panel").forEach((panel) => {
        panel.classList.toggle("is-active", panel.id === `panel-${tab.dataset.tab}`);
      });
    });
  });

  /* ---------- top-level render ---------- */

  function renderAll() {
    els.drop.hidden = true;
    els.app.hidden = false;
    els.badge.hidden = false;
    els.badge.textContent = state.lineage.artifact_type || "phylodigy.architecture_lineage";
    els.digest.textContent = `lineage digest sha256:${state.lineage.digest || "?"}`;
    renderTree();
    renderDiagnostics();
    renderMatrix();
    renderGenomes();
    renderEvidence();
    els.raw.textContent = JSON.stringify(state.lineage, null, 2);
  }

  /* ---------- tree panel ---------- */

  document.querySelectorAll(".layout-toggle").forEach((button) => {
    button.addEventListener("click", () => {
      state.layoutMode = button.dataset.layout;
      document.querySelectorAll(".layout-toggle").forEach((b) => {
        const active = b === button;
        b.classList.toggle("is-active", active);
        b.setAttribute("aria-pressed", String(active));
      });
      renderTree();
    });
  });

  function renderTree() {
    if (state.layoutMode === "circle") renderTreeRadial();
    else renderTreeRect();
  }

  function renderTreeRect() {
    const tree = state.lineage.tree;
    const layout = L.layoutTree(tree, { rowHeight: 30 });
    state._layout = layout;
    const labels = L.tipLabels(tree, Object.fromEntries(state.names));
    const traits = L.edgeTraits(tree, state.lineage.character_matrix);

    const leftPad = 20, topPad = 14;
    const scale = Math.min(9, 540 / Math.max(layout.maxX, 1e-9));
    const px = (x) => leftPad + x * scale;
    const longestTip = Math.max(...[...labels.entries()]
      .filter(([id]) => !L.isInternal(id))
      .map(([, name]) => name.length), 1);
    const width = px(layout.maxX) + 10 + longestTip * 7.2 + 24;
    const maxY = Math.max(...[...layout.nodes.values()].map((n) => n.y), 0);
    const height = maxY + topPad + 26;

    const svg = els.svg;
    svg.textContent = "";
    svg.setAttribute("width", Math.ceil(width));
    svg.setAttribute("height", Math.ceil(height));
    svg.setAttribute("viewBox", `0 0 ${Math.ceil(width)} ${Math.ceil(height)}`);

    drawEdgesRect(svg, px, topPad, layout, tree, traits);
    for (const node of layout.nodes.values()) {
      drawNode(svg, (id) => px(layout.nodes.get(id).x), topPad, node, labels, layout, "rect");
    }

    els.scale.textContent =
      `rectangular phylogram — horizontal distance = cumulative graph-character branch length ` +
      `(≈${formatScale(1 / scale)} characters/px). Root marker = final NJ join; the inferred backbone itself is unrooted. ` +
      `Chips on branches are gained (+n, green) or lost (−n, red) graph characters.`;
  }

  function formatScale(value) {
    if (value >= 0.01) return String(Math.round(value * 100) / 100);
    return value.toExponential(2);
  }

  function renderTreeRadial() {
    const tree = state.lineage.tree;
    const labels = L.tipLabels(tree, Object.fromEntries(state.names));
    const traits = L.edgeTraits(tree, state.lineage.character_matrix);
    // Angle gap so tip labels have room between wedges.
    const layout = L.layoutRadial(tree, { arcGap: 0.06 });
    state._layout = layout;

    const radialScale = 26; // px per branch-length unit
    const labelPad = Math.max(...[...labels.entries()]
      .filter(([id]) => !L.isInternal(id))
      .map(([, name]) => name.length), 1) * 7.0 + 26;
    const R = layout.maxX * radialScale + labelPad;
    const size = 2 * R + 30;
    const cx = size / 2, cy = size / 2;
    const toXY = (node) => [cx + node.radius * radialScale * Math.sin(node.angle),
                            cy - node.radius * radialScale * Math.cos(node.angle)];

    const svg = els.svg;
    svg.textContent = "";
    svg.setAttribute("width", Math.ceil(size));
    svg.setAttribute("height", Math.ceil(size));
    svg.setAttribute("viewBox", `0 0 ${Math.ceil(size)} ${Math.ceil(size)}`);

    // Reference rings at each unit of cumulative branch length.
    const rings = document.createElementNS(NS, "g");
    for (let r = radialScale; r <= layout.maxX * radialScale + 1; r += radialScale) {
      const ring = document.createElementNS(NS, "circle");
      ring.setAttribute("cx", cx); ring.setAttribute("cy", cy); ring.setAttribute("r", r);
      ring.setAttribute("fill", "none");
      ring.setAttribute("stroke", "var(--border, #d8dee6)");
      ring.setAttribute("stroke-dasharray", "2 4");
      ring.setAttribute("stroke-width", "0.6");
      rings.appendChild(ring);
    }
    svg.appendChild(rings);

    // Edges: radial spoke from parent radius to child radius at the child's
    // angle, then an arc at the parent's radius spanning the parent's children.
    const edgeLayer = document.createElementNS(NS, "g");
    for (const edge of tree.edges) {
      const parent = layout.nodes.get(edge.parent);
      const child = layout.nodes.get(edge.child);
      const [x0, y0] = toXY(parent), [x1, y1] = toXY(child);
      const line = document.createElementNS(NS, "line");
      line.setAttribute("x1", x0); line.setAttribute("y1", y0);
      line.setAttribute("x2", x1); line.setAttribute("y2", y1);
      line.setAttribute("class", edge.length_clamped ? "branch branch-clamped" : "branch");
      edgeLayer.appendChild(line);
      // Traits on this edge, at the spoke midpoint.
      const t = traits.get(edge.child);
      if (t && (t.gained.length || t.lost.length)) {
        drawTraitChips(edgeLayer, (x0 + x1) / 2, (y0 + y1) / 2, t, edge);
      }
    }
    // Arcs connecting each internal node's children at the internal radius.
    const childMap = L.buildTreeIndex(tree).children;
    for (const [parentId, kids] of childMap) {
      if (kids.length < 2) continue;
      const parent = layout.nodes.get(parentId);
      const r = parent.radius * radialScale;
      const angles = kids.map((k) => layout.nodes.get(k).angle).sort((a, b) => a - b);
      const a0 = angles[0], a1 = angles[angles.length - 1];
      const arc = document.createElementNS(NS, "path");
      const [ax0, ay0] = [cx + r * Math.sin(a0), cy - r * Math.cos(a0)];
      const [ax1, ay1] = [cx + r * Math.sin(a1), cy - r * Math.cos(a1)];
      arc.setAttribute("d", `M ${ax0} ${ay0} A ${r} ${r} 0 0 1 ${ax1} ${ay1}`);
      arc.setAttribute("class", "branch");
      edgeLayer.appendChild(arc);
    }
    svg.appendChild(edgeLayer);

    // Nodes + labels (rotated tip labels along their spoke).
    for (const node of layout.nodes.values()) {
      drawNodeRadial(svg, cx, cy, radialScale, node, labels, layout);
    }

    els.scale.textContent =
      `circular phylogram — radial distance from the center = cumulative graph-character branch length ` +
      `(dashed rings at 1-character intervals). Root marker = final NJ join; the inferred backbone itself is unrooted. ` +
      `Chips on branches are gained (+n, green) or lost (−n, red) graph characters.`;
  }

  function drawTraitChips(layer, x, y, traits, edge) {
    const chips = [];
    if (traits.gained.length) chips.push({ kind: "gained", text: `+${traits.gained.length}`, ids: traits.gained });
    if (traits.lost.length) chips.push({ kind: "lost", text: `−${traits.lost.length}`, ids: traits.lost });
    let dx = -((chips.length - 1) * 16);
    for (const chip of chips) {
      const g = document.createElementNS(NS, "g");
      g.setAttribute("class", `trait-chip trait-${chip.kind}`);
      const rect = document.createElementNS(NS, "rect");
      const w = chip.text.length * 6.4 + 8, h = 15;
      rect.setAttribute("x", x + dx - w / 2);
      rect.setAttribute("y", y - h / 2 - 10);
      rect.setAttribute("width", w); rect.setAttribute("height", h);
      const text = document.createElementNS(NS, "text");
      text.setAttribute("x", x + dx);
      text.setAttribute("y", y - 10 + h / 2 - 3.5);
      text.setAttribute("text-anchor", "middle");
      text.textContent = chip.text;
      g.append(rect, text);
      g.addEventListener("click", (event) => {
        event.stopPropagation();
        showTraitPopover(event.clientX, event.clientY, chip, edge);
      });
      layer.appendChild(g);
      dx += 34;
    }
  }

  function drawEdgesRect(svg, px, topPad, layout, tree, traits) {
    const g = document.createElementNS(NS, "g");
    for (const edge of tree.edges) {
      const parent = layout.nodes.get(edge.parent);
      const child = layout.nodes.get(edge.child);
      if (!parent || !child) continue;
      const x0 = px(parent.x), x1 = px(child.x);
      const y0 = topPad + parent.y, y1 = topPad + child.y;
      const path = document.createElementNS(NS, "path");
      path.setAttribute("d", `M ${x0} ${y0} L ${x0} ${y1} L ${x1} ${y1}`);
      path.setAttribute("class", edge.length_clamped ? "branch branch-clamped" : "branch");
      g.appendChild(path);
      // Branch length caption above long-enough segments, and always for clamped limbs.
      if (edge.length_clamped || x1 - x0 > 46) {
        const label = document.createElementNS(NS, "text");
        label.setAttribute("x", (x0 + x1) / 2);
        label.setAttribute("y", y1 - 14);
        label.setAttribute("text-anchor", "middle");
        label.setAttribute("class", "length-label");
        label.textContent = edge.length_clamped
          ? `raw ${L.formatLength(edge.raw_branch_length)} → clamped to 0`
          : L.formatLength(edge.branch_length);
        g.appendChild(label);
      }
      const t = traits.get(edge.child);
      if (t && (t.gained.length || t.lost.length)) {
        drawTraitChips(g, (x0 + x1) / 2, y1, t, edge);
      }
    }
    svg.appendChild(g);
  }

  function drawNodeRadial(svg, cx, cy, radialScale, node, labels, layout) {
    const x = cx + node.radius * radialScale * Math.sin(node.angle);
    const y = cy - node.radius * radialScale * Math.cos(node.angle);
    const g = document.createElementNS(NS, "g");
    g.setAttribute("class", "hoverable" + (state.selected === node.id ? " selected" : ""));
    g.setAttribute("tabindex", "0");
    g.setAttribute("role", "button");
    g.setAttribute("aria-label", `Inspect ${labels.get(node.id) || node.id}`);
    g.setAttribute("data-node-id", node.id);

    const dot = document.createElementNS(NS, "circle");
    dot.setAttribute("cx", x); dot.setAttribute("cy", y);
    dot.setAttribute("r", node.internal ? 5 : 6);
    dot.setAttribute("class", "node-dot" + (node.internal ? " internal" : ""));
    g.appendChild(dot);

    const label = document.createElementNS(NS, "text");
    label.textContent = labels.get(node.id) || node.id;
    if (node.internal) {
      // Keep internal labels inside — place toward the root along the spoke.
      label.setAttribute("x", (x + cx) / 2);
      label.setAttribute("y", (y + cy) / 2 - 6);
      label.setAttribute("class", "internal-label");
      label.setAttribute("text-anchor", "middle");
    } else {
      // Rotate the tip label to read outward along its spoke.
      const deg = (node.angle * 180) / Math.PI;
      const flip = deg > 90 && deg < 270;
      const lx = x + 10 * Math.sin(node.angle);
      const ly = y - 10 * Math.cos(node.angle);
      label.setAttribute("x", lx);
      label.setAttribute("y", ly);
      label.setAttribute("class", "tip-label");
      label.setAttribute("text-anchor", flip ? "end" : "start");
      label.setAttribute("transform", `rotate(${flip ? deg + 180 : deg}, ${lx}, ${ly})`);
      label.setAttribute("dy", "0.35em");
    }
    g.appendChild(label);

    attachNodeHandlers(g, node, layout, [x, y]);
    svg.appendChild(g);
  }

  // Shared node interaction wiring. rectXY positions the hit rect.
  function attachNodeHandlers(g, node, layout, rectXY) {
    const hit = document.createElementNS(NS, "rect");
    hit.setAttribute("x", rectXY[0] - 24);
    hit.setAttribute("y", rectXY[1] - 13);
    hit.setAttribute("width", 48);
    hit.setAttribute("height", 26);
    hit.setAttribute("fill", "#000");
    hit.setAttribute("fill-opacity", "0");
    hit.setAttribute("pointer-events", "all");
    hit.setAttribute("class", "hit-rect");
    g.appendChild(hit);
    hit.addEventListener("click", () => selectNode(node.id, layout));
    g.addEventListener("click", () => selectNode(node.id, layout));
    g.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        selectNode(node.id, layout);
      }
    });
  }

  /* ---------- trait popover ---------- */

  let popoverEl = null;

  function showTraitPopover(x, y, chip, edge) {
    hideTraitPopover();
    const el = document.createElement("div");
    el.className = "trait-popover";
    const h = document.createElement("h4");
    h.textContent = `${chip.kind === "gained" ? "Gained" : "Lost"} graph characters — edge → ${edge.child}`;
    el.appendChild(h);
    const list = document.createElement("ul");
    for (const id of chip.ids) {
      const li = document.createElement("li");
      li.textContent = id;
      list.appendChild(li);
    }
    el.appendChild(list);
    const note = document.createElement("p");
    note.className = "pop-note";
    note.textContent =
      "Anonymous graph-derived characters (content digests), placed on branches by " +
      "parsimony intersection over reconstructed ancestor states. Candidate signal, " +
      "not an ancestral gain/loss claim or a named technique.";
    el.appendChild(note);
    document.body.appendChild(el);
    const px = Math.min(x + 12, window.innerWidth - 360);
    const py = Math.min(y + 12, window.innerHeight - 220);
    el.style.left = `${Math.max(8, px)}px`;
    el.style.top = `${Math.max(8, py)}px`;
    popoverEl = el;
    setTimeout(() => document.addEventListener("mousedown", onOutside), 0);
  }

  function onOutside(event) {
    if (popoverEl && !popoverEl.contains(event.target)) hideTraitPopover();
  }

  function hideTraitPopover() {
    if (popoverEl) popoverEl.remove();
    popoverEl = null;
    document.removeEventListener("mousedown", onOutside);
  }


  function drawNode(svg, pxOf, topPad, node, labels, layout, mode) {
    const g = document.createElementNS(NS, "g");
    g.setAttribute("class", "hoverable" + (state.selected === node.id ? " selected" : ""));
    // Expose as a focusable, named control so assistive tech and automated
    // drivers can reach it (plain SVG <g> is otherwise inert).
    g.setAttribute("tabindex", "0");
    g.setAttribute("role", "button");
    g.setAttribute("aria-label", `Inspect ${labels.get(node.id) || node.id}`);
    g.setAttribute("data-node-id", node.id);
    const cx = pxOf(node.id), cy = topPad + node.y;

    const dot = document.createElementNS(NS, "circle");
    dot.setAttribute("cx", cx);
    dot.setAttribute("cy", cy);
    dot.setAttribute("r", node.internal ? 5 : 6);
    dot.setAttribute("class", "node-dot" + (node.internal ? " internal" : ""));
    g.appendChild(dot);

    const label = document.createElementNS(NS, "text");
    label.textContent = labels.get(node.id) || node.id;
    if (node.internal) {
      label.setAttribute("x", cx - 8);
      label.setAttribute("y", cy - 9);
      label.setAttribute("class", "internal-label");
      label.setAttribute("text-anchor", "end");
    } else {
      label.setAttribute("x", cx + 10);
      label.setAttribute("y", cy + 4);
      label.setAttribute("class", "tip-label");
    }
    g.appendChild(label);

    attachNodeHandlers(g, node, layout, [cx, cy]);
    svg.appendChild(g);
  }

  function selectNode(id, layout) {
    state.selected = state.selected === id ? null : id;
    // Update highlight classes in place (no SVG rebuild mid-click), then
    // fill the detail panel for a current selection.
    svgRoot().querySelectorAll("g[data-node-id]").forEach((g) => {
      g.classList.toggle("selected", g.getAttribute("data-node-id") === state.selected);
    });
    if (!state.selected) {
      els.detail.hidden = true;
      return;
    }
    const tree = state.lineage.tree;
    const edge = tree.edges.find((e) => e.child === id);
    const genome = (state.lineage.genomes || []).find((g) => g.id === id);
    const internal = L.isInternal(id);

    els.detail.textContent = "";
    const title = document.createElement("h2");
    title.textContent = internal
      ? `Hypothetical ancestor ${id.split(":").pop()}`
      : id;
    const dl = document.createElement("dl");
    addDetail(dl, "kind", internal ? "internal (inferred, unnamed)" : "tip (observed artifact)");
    if (internal) {
      const node = layout.nodes.get(id);
      addDetail(dl, "descends to", (node.childIds || []).join(", "));
    }
    if (edge) {
      addDetail(dl, "parent", edge.parent);
      addDetail(dl, "branch length", `${L.formatLength(edge.branch_length)} (raw ${L.formatLength(edge.raw_branch_length)}${edge.length_clamped ? ", clamped" : ""})`);
    } else {
      addDetail(dl, "parent", "— (final NJ join; display root)");
    }
    if (genome) {
      addDetail(dl, "date", genome.date_min === genome.date_max ? genome.date_max : `${genome.date_min} → ${genome.date_max}`);
      addDetail(dl, "genome digest", genome.digest);
      addDetail(dl, "graph digest", genome.graph_digest);
    }
    const counts = (state.lineage.character_matrix || {}).counts_by_artifact || {};
    if (counts[id]) {
      const total = counts[id].reduce((sum, n) => sum + n, 0);
      addDetail(dl, "character vector", `${counts[id].length} characters, mass ${total}`);
    }
    els.detail.append(title, dl);
    els.detail.hidden = false;
  }

  function addDetail(dl, term, value) {
    const dt = document.createElement("dt");
    dt.textContent = term;
    const dd = document.createElement("dd");
    dd.textContent = value == null ? "—" : String(value);
    dl.append(dt, dd);
  }

  function svgRoot() {
    return els.svg;
  }

  // Surface uncaught render errors in the footer instead of dying silently —
  // a thrown render must never leave stale geometry on screen.
  window.addEventListener("error", (event) => {
    els.digest.textContent = `render error: ${event.message}`;
    els.digest.style.color = "var(--bad, #b91c1c)";
    els.scale.textContent = `render error: ${event.message}`;
  });

  /* ---------- diagnostics ---------- */

  function renderDiagnostics() {
    const d = state.lineage.tree.tree_likeness || {};
    const nj = d.neighbor_joining || {};
    const rows = [
      ["tree metric within tolerance", bool(d.tree_metric_within_tolerance)],
      ["additive (four-point) within tolerance", bool(d.additive_within_tolerance)],
      ["metric (triangle) within tolerance", bool(d.metric_within_tolerance)],
      ["distance domain valid", bool(d.distance_domain_valid)],
      ["distance pairs", d.distance_pair_count],
      ["quartets checked", d.quartet_count],
      ["quartet violations", d.violation_count],
      ["triangle violations", d.triangle_violation_count],
      ["negative limbs", nj.negative_limb_count],
      ["negative limbs beyond tolerance", nj.negative_limb_count_beyond_tolerance],
      ["min raw branch length", L.formatLength(nj.minimum_raw_branch_length)],
      ["method", state.lineage.tree.method],
      ["distance model", state.lineage.tree.distance_model],
    ];
    els.diagnostics.textContent = "";
    for (const [key, value] of rows) {
      if (value === undefined) continue;
      const div = document.createElement("div");
      const dt = document.createElement("dt");
      dt.textContent = key;
      const dd = document.createElement("dd");
      if (typeof value === "boolean") {
        dd.textContent = value ? "true" : "false";
        dd.className = value ? "diag-good" : "diag-bad";
      } else {
        dd.textContent = String(value);
      }
      div.append(dt, dd);
      els.diagnostics.appendChild(div);
    }

    const signals = [];
    for (const v of d.triangle_violations || []) {
      signals.push(`triangle ${v.artifact_ids.join(" / ")} via ${v.via}: +${L.formatLength(v.violation)}`);
    }
    for (const v of d.violations || []) {
      signals.push(`quartet ${v.artifact_ids.join(" / ")}: ${L.formatLength(v.violation)}`);
    }
    for (const limb of nj.negative_limbs || []) {
      signals.push(`negative limb ${limb.parent} → ${limb.child}: raw ${L.formatLength(limb.raw_branch_length)}`);
    }
    els.violationPanel.hidden = signals.length === 0;
    els.violationList.textContent = "";
    for (const signal of signals) {
      const li = document.createElement("li");
      li.textContent = signal;
      els.violationList.appendChild(li);
    }
  }

  function bool(value) { return Boolean(value); }

  /* ---------- distance matrix ---------- */

  function renderMatrix() {
    const taxa = state.lineage.tree.taxa;
    const rows = L.matrixRows(taxa, state.lineage.comparisons);
    const max = Math.max(...rows.flat().map((c) => c.distance || 0));
    const table = els.matrix;
    table.textContent = "";
    const head = document.createElement("tr");
    head.appendChild(document.createElement("th"));
    for (const id of taxa) {
      const th = document.createElement("th");
      th.className = "col-header";
      th.textContent = id;
      head.appendChild(th);
    }
    table.appendChild(head);
    rows.forEach((cells, i) => {
      const tr = document.createElement("tr");
      const th = document.createElement("th");
      th.textContent = taxa[i];
      tr.appendChild(th);
      for (const cell of cells) {
        const td = document.createElement("td");
        if (cell.isSelf) {
          td.textContent = "—";
          td.className = "self";
        } else {
          td.textContent = L.formatLength(cell.distance);
          td.style.background = L.heatColor(cell.distance, max);
          td.title = `${cell.left} ↔ ${cell.right}: ${cell.distance}`;
        }
        tr.appendChild(td);
      }
      table.appendChild(tr);
    });
  }

  /* ---------- genomes ---------- */

  function renderGenomes() {
    const table = els.genomes;
    table.textContent = "";
    const header = document.createElement("tr");
    for (const key of ["artifact id", "date range", "genome digest", "graph digest"]) {
      const th = document.createElement("th");
      th.textContent = key;
      header.appendChild(th);
    }
    table.appendChild(header);
    for (const genome of state.lineage.genomes || []) {
      const tr = document.createElement("tr");
      const id = cell(genome.id, false);
      const dates = cell(
        genome.date_min === genome.date_max
          ? genome.date_max || "—"
          : `${genome.date_min || "?"} → ${genome.date_max || "?"}`,
        false
      );
      const digest = cell(L.digestShort(genome.digest), true);
      digest.title = genome.digest;
      const graph = cell(L.digestShort(genome.graph_digest), true);
      graph.title = genome.graph_digest;
      tr.append(id, dates, digest, graph);
      table.appendChild(tr);
    }
  }

  function cell(text, mono) {
    const td = document.createElement("td");
    td.textContent = text == null ? "—" : String(text);
    if (mono) td.className = "mono";
    return td;
  }

  /* ---------- evidence ---------- */

  function renderEvidence() {
    const boundary = state.lineage.structural_evidence_boundary || {};
    els.evidenceBoundary.innerHTML = "";
    const strong = document.createElement("strong");
    strong.textContent = "Structural evidence boundary";
    const p = document.createElement("div");
    const paper = String(boundary.paper_can_create_characters);
    p.innerHTML =
      `Structural source: <code>${escapeHtml(boundary.structural_source || "operator_layer_graph")}</code> · ` +
      `paper annotations can create characters: <code>${escapeHtml(paper)}</code>. ` +
      `External records below are retained beside the tree, never inside its distances.`;
    els.evidenceBoundary.append(strong, p);

    const records = state.lineage.external_evidence || [];
    els.evidenceEmpty.hidden = records.length !== 0;
    els.evidenceTable.hidden = records.length === 0;
    if (!records.length) return;
    const keys = [...new Set(records.flatMap((r) => Object.keys(r)))].sort();
    els.evidenceTable.textContent = "";
    const head = document.createElement("tr");
    for (const key of keys) {
      const th = document.createElement("th");
      th.textContent = key;
      head.appendChild(th);
    }
    els.evidenceTable.appendChild(head);
    for (const record of records) {
      const tr = document.createElement("tr");
      for (const key of keys) {
        const value = record[key];
        const text = value == null ? "—" : typeof value === "object" ? JSON.stringify(value) : String(value);
        tr.appendChild(cell(text, true));
      }
      els.evidenceTable.appendChild(tr);
    }
  }

  function escapeHtml(text) {
    return text.replace(/[&<>"']/g, (ch) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[ch]));
  }

  /* ---------- boot ---------- */
  loadSample(false);
})();
