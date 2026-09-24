"use strict";

/* Pure layout and formatting logic for the phylodigital tree viewer.
 *
 * This module is DOM-free. It runs in the browser (loaded as a plain
 * script, attaching `PhylodigyLayout` to globalThis) and under Node (via
 * module.exports) so `node --test` can pin its behavior without a browser.
 *
 * The artifact shape consumed here is the canonical
 * `phylodigy.architecture_lineage` record produced by
 * `phylodigy infer-lineage` / `phylodigy toy-tree`: an unrooted
 * neighbor-joining edge list with `parent`/`child`/`branch_length`/
 * `raw_branch_length`/`length_clamped`, plus `tree_likeness` diagnostics,
 * pairwise `comparisons`, and `genomes` metadata.
 */

(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory();
  } else {
    root.PhylodigyLayout = factory();
  }
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  const INTERNAL_PREFIX = "__phylodigy_internal__:";

  function isInternal(id) {
    return typeof id === "string" && id.startsWith(INTERNAL_PREFIX);
  }

  /* Ordered postorder and child access shared by the new passes. */
  function postOrder(tree, rootId) {
    const { children } = buildTreeIndex(tree);
    const order = [];
    (function visit(id) {
      for (const child of children.get(id) || []) visit(child);
      order.push(id);
    })(rootId || rootTree(tree));
    return { order, children };
  }

  /* Index an edge list into parent -> ordered children and child -> edge
   * lookups, and the full set of node ids. Children keep the edge-list
   * order, which the producer sorts lexically — deterministic layout. */
  function buildTreeIndex(tree) {
    if (!tree || !Array.isArray(tree.edges)) {
      throw new TypeError("tree.edges must be an array");
    }
    const children = new Map();
    const edgeByChild = new Map();
    const nodeIds = new Set();
    for (const edge of tree.edges) {
      if (!children.has(edge.parent)) children.set(edge.parent, []);
      children.get(edge.parent).push(edge.child);
      edgeByChild.set(edge.child, edge);
      nodeIds.add(edge.parent);
      nodeIds.add(edge.child);
    }
    return { children, edgeByChild, nodeIds };
  }

  /* The unrooted neighbor-joining backbone is displayed rooted at the
   * final join: the node that never appears as a `child`. */
  function rootTree(tree) {
    const { nodeIds, edgeByChild } = buildTreeIndex(tree);
    const roots = [...nodeIds].filter((id) => !edgeByChild.has(id));
    if (roots.length !== 1) {
      throw new Error(
        `expected exactly one tree root, found ${roots.length || "none"}`
      );
    }
    return roots[0];
  }

  /* Phylogram layout: x is the cumulative clampled branch length from the
   * root, y is a tip row. Tips receive one row each in first-visit order;
   * internal nodes sit at the mean row of their children. */
  function layoutTree(tree, options) {
    const opts = options || {};
    const rowHeight = opts.rowHeight || 24;
    const index = buildTreeIndex(tree);
    const root = rootTree(tree);
    const nodes = new Map();
    const tips = [];
    let nextRow = 0;

    function place(id, x) {
      const children = index.children.get(id) || [];
      let y;
      if (children.length === 0) {
        nextRow += 1;
        y = nextRow * rowHeight;
        tips.push({ id, row: nextRow - 1, y });
        nodes.set(id, { id, x, y, row: nextRow - 1, internal: false });
      } else {
        const ys = children.map((child) => {
          const edge = index.edgeByChild.get(child);
          return place(child, x + Math.max(0, edge.branch_length));
        });
        y = ys.reduce((sum, value) => sum + value, 0) / ys.length;
        nodes.set(id, {
          id,
          x,
          y,
          internal: true,
          childIds: children.slice(),
        });
      }
      return y;
    }

    place(root, 0);
    const maxX = Math.max(
      ...[...nodes.values()].map((node) => node.x),
      0
    );
    return { nodes, tips, rootId: root, maxX, rowHeight, tipCount: tips.length };
  }

  /* Radial phylogram: same cumulative branch lengths as layoutTree, with tip
   * rows wrapped evenly around a circle. Angles in radians, 0 = 12 o'clock,
   * clockwise. Each node carries polar (radius, angle) and a Cartesian
   * projection (x, y) for direct SVG placement. */
  function layoutRadial(tree, options) {
    const opts = options || {};
    const gap = opts.arcGap === undefined ? 0.0 : opts.arcGap;
    const base = layoutTree(tree, opts);
    const nodes = new Map();
    const tips = [];
    const n = base.tips.length;
    const step = (2 * Math.PI - gap * n) / Math.max(n, 1);
    const angleFor = new Map(
      base.tips.map((tip, i) => [tip.id, i * (step + gap)])
    );

    const { order, children } = postOrder(tree, base.rootId);
    let maxX = 0;
    for (const id of order) {
      const node = base.nodes.get(id);
      maxX = Math.max(maxX, node.x);
      const kids = children.get(id) || [];
      let angle;
      if (kids.length === 0) {
        angle = angleFor.get(id);
      } else {
        angle =
          kids.reduce((sum, child) => sum + nodes.get(child).angle, 0) /
          kids.length;
      }
      const point = {
        id,
        radius: node.x,
        angle,
        internal: node.internal,
        // 12-o'clock start, clockwise.
        x: node.x * Math.sin(angle),
        y: -node.x * Math.cos(angle),
      };
      nodes.set(id, point);
      if (kids.length === 0) tips.push(point);
    }
    for (const id of angleFor.keys()) {
      // ensure every tip made it into the postorder (guards malformed trees)
      if (!nodes.has(id)) throw new Error(`tip missing from layout: ${id}`);
    }
    return {
      nodes,
      tips,
      rootId: base.rootId,
      maxX,
      tipCount: tips.length,
      step,
      tipNodes: tips,
    };
  }

  /* Reconstruct per-node character states. Tip states come verbatim from the
   * character matrix; an internal node carries exactly the characters every
   * one of its descendants carries (the parsimonious intersection), at the
   * minimum shared count. This deliberately stays in the project's
   * "characters are anonymous, gains/losses are candidate claims" register:
   * the intersection is reconstructible from the matrix alone. */
  function reconstructStates(tree, characterMatrix) {
    const characterIds = characterMatrix.character_ids || [];
    const countsByArtifact = characterMatrix.counts_by_artifact || {};
    const representatives = characterMatrix.artifact_representatives || {};
    const { order, children } = postOrder(tree);
    const states = new Map();
    const tipStates = new Map();

    const tipState = (id) => {
      const representative = representatives[id] || id;
      // Compact lineages may have many leaves backed by one row. Reuse one
      // immutable state object for all aliases instead of expanding each row.
      const cacheKey = Object.prototype.hasOwnProperty.call(countsByArtifact, id)
        ? id
        : representative;
      if (tipStates.has(cacheKey)) return tipStates.get(cacheKey);
      const row = countsByArtifact[id] || countsByArtifact[representative] || [];
      const state = {};
      characterIds.forEach((characterId, i) => {
        const count = row[i];
        if (typeof count === "number" && count > 0) state[characterId] = count;
      });
      Object.freeze(state);
      tipStates.set(cacheKey, state);
      return state;
    };

    for (const id of order) {
      const kids = children.get(id) || [];
      if (kids.length === 0) {
        states.set(id, tipState(id));
        continue;
      }
      const childStates = kids.map((child) => states.get(child));
      if (childStates.every((state) => state === childStates[0])) {
        states.set(id, childStates[0]);
        continue;
      }
      const shared = {};
      for (const characterId of Object.keys(childStates[0])) {
        let count = childStates[0][characterId];
        let presentInAll = true;
        for (let i = 1; i < childStates.length; i++) {
          const other = childStates[i][characterId];
          if (typeof other !== "number" || other <= 0) {
            presentInAll = false;
            break;
          }
          count = Math.min(count, other);
        }
        if (presentInAll) shared[characterId] = count;
      }
      states.set(id, Object.freeze(shared));
    }
    return states;
  }

  /* Per-edge traits: which characters a node gained or lost relative to its
   * parent's reconstructed state. Keyed by child id; the display root has no
   * incoming edge and is absent. Count-based: gained/lost compare counts, so
   * an increase from 1 to 2 reads as a gain of one copy. */
  function edgeTraits(tree, characterMatrix) {
    const states = reconstructStates(tree, characterMatrix);
    const result = new Map();
    for (const edge of tree.edges) {
      const parent = states.get(edge.parent) || {};
      const child = states.get(edge.child) || {};
      const gained = [];
      const lost = [];
      for (const [id, count] of Object.entries(child)) {
        const delta = count - (parent[id] || 0);
        if (delta > 0) gained.push(id);
      }
      for (const [id, count] of Object.entries(parent)) {
        const delta = count - (child[id] || 0);
        if (delta > 0) lost.push(id);
      }
      gained.sort();
      lost.sort();
      result.set(edge.child, { gained, lost });
    }
    return result;
  }

  /* Display labels: names come from profile metadata when available,
   * synthetic internal nodes become explicit hypothetical ancestors. */
  function tipLabels(tree, names) {
    const lookup = names || {};
    const labels = new Map();
    const { nodeIds } = buildTreeIndex(tree);
    const internals = [...nodeIds]
      .filter(isInternal)
      .sort();
    let n = 0;
    for (const id of internals) {
      labels.set(id, `hypothetical ancestor ${n++}`);
    }
    for (const id of nodeIds) {
      if (!labels.has(id)) labels.set(id, lookup[id] || id);
    }
    return labels;
  }

  /* Pairwise distances indexed by the unordered artifact-id pair. */
  function distanceLookup(comparisons, artifactRepresentatives) {
    // Passing a whole lineage enables compact representative expansion while
    // preserving the historical comparisons-array calling convention.
    if (!Array.isArray(comparisons) && comparisons && typeof comparisons === "object") {
      artifactRepresentatives = (comparisons.character_matrix || {}).artifact_representatives || {};
      comparisons = comparisons.comparisons;
    }
    const representatives = artifactRepresentatives || {};
    const distances = new Map();
    for (const item of comparisons || []) {
      const pair = JSON.stringify([item.left.id, item.right.id].sort());
      distances.set(pair, item.graph_distance.distance);
    }
    return function (left, right) {
      if (left === right) return 0;
      const leftRep = representatives[left] || left;
      const rightRep = representatives[right] || right;
      if (leftRep === rightRep) return 0;
      return distances.get(JSON.stringify([leftRep, rightRep].sort()));
    };
  }

  /* Full n×n distance matrix rows for rendering, preserving taxa order. */
  function matrixRows(taxa, lineageOrComparisons, artifactRepresentatives) {
    const lookup = distanceLookup(lineageOrComparisons, artifactRepresentatives);
    return taxa.map((leftId) =>
      taxa.map((rightId) => ({
        left: leftId,
        right: rightId,
        distance: lookup(leftId, rightId),
        isSelf: leftId === rightId,
      }))
    );
  }

  /* Cell background for the matrix heat scale: white -> accent blue. */
  function heatColor(value, max) {
    if (value === undefined || !isFinite(value) || max <= 0) {
      return "rgba(148, 163, 184, 0.15)";
    }
    const t = Math.min(1, value / max);
    const light = 96 - t * 42; // 96% -> 54% lightness
    return `hsl(221, 72%, ${light}%)`;
  }

  function digestShort(digest, count) {
    const n = count || 12;
    if (typeof digest !== "string") return "";
    return digest.length > n ? `${digest.slice(0, n)}…` : digest;
  }

  function formatLength(value) {
    if (typeof value !== "number" || !isFinite(value)) return "?";
    if (value === 0) return "0";
    if (Math.abs(value) >= 0.01) return String(Math.round(value * 100) / 100);
    return value.toExponential(2);
  }

  /* artifact-id -> display name; this project keeps names out of
   * structure, so ids are the fallback rather than an error. */
  function genomeNames(lineage) {
    const names = new Map();
    for (const genome of lineage.genomes || []) {
      names.set(genome.id, genome.id);
    }
    return names;
  }

  return {
    INTERNAL_PREFIX,
    buildTreeIndex,
    digestShort,
    distanceLookup,
    edgeTraits,
    formatLength,
    genomeNames,
    heatColor,
    isInternal,
    layoutRadial,
    layoutTree,
    matrixRows,
    postOrder,
    reconstructStates,
    rootTree,
    tipLabels,
  };
});
