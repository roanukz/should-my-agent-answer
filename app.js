/* ===========================================================================
   The findings explorer.

   No framework, no build step, no network. The data arrives as a global from
   data/site.js, which is a <script src> rather than a fetch() so the page works
   when it is opened straight off disk.

   The "how this was computed" toggle on every card is the point of the tool. A
   finding you cannot audit is a finding nobody acts on, so the graph path and
   the verbatim evidence span behind each hop are always one click away, and the
   click is a native <details> so it still works if this script never runs.
   =========================================================================== */

(function () {
  'use strict';

  var DATA = window.SMAA || { findings: [], counts: {}, sections: {}, manifest: {} };
  var PAGE_SIZE = 25;

  var TYPE_LABEL = {
    near_miss: 'Near miss',
    orphan_concept: 'Orphan concept',
    retrieval_collision: 'Retrieval collision'
  };

  var TYPE_BLURB = {
    near_miss:
      'A concept people ask about, that a section’s instructions depend on, ' +
      'and that no section in the corpus explains.',
    orphan_concept:
      'A concept the documentation leans on or refers to and never explains. ' +
      'The same shape as a near miss, without anyone asking yet.',
    retrieval_collision:
      'Two sections a single question maps to with comparable concept overlap, ' +
      'where only one carries the answer.'
  };

  /* The four the select offers. Also used to validate a sort read out of the
     address bar, so an unknown value falls back rather than silently sorting
     by nothing. */
  var SORTS = { demand: 1, 'demand-asc': 1, concept: 1, id: 1 };

  var state = { type: 'all', sort: 'demand', query: '', shown: PAGE_SIZE };

  var els = {};

  /* ---------- small helpers ---------- */

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  function frag() {
    return document.createDocumentFragment();
  }

  /* Turn a repository path plus a line range into a link at the pinned commit,
     so a reader can check the span against the real file rather than trusting
     the quote on the card. */
  function sourceLink(path, lines) {
    if (!path) return null;
    var base = DATA.source_base;
    var href;
    if (path.indexOf('data/raw/discussions/') === 0) {
      var number = path.split('/').pop().replace(/\.md$/, '');
      href = 'https://github.com/fastapi/fastapi/discussions/' + number;
    } else {
      href = base + '/' + path;
      if (lines && lines.length === 2) {
        href += '#L' + lines[0] + (lines[1] !== lines[0] ? '-L' + lines[1] : '');
      }
    }
    var a = el('a', null, path + (lines ? ':' + lines[0] + (lines[1] !== lines[0] ? '-' + lines[1] : '') : ''));
    a.href = href;
    a.rel = 'noopener';
    a.target = '_blank';
    return a;
  }

  function haystack(finding) {
    var parts = [
      finding.concept_label,
      finding.missing,
      finding.doc_title,
      finding.section_heading,
      finding.id,
      finding.concept_kind
    ];
    (finding.question_titles || []).forEach(function (t) { parts.push(t); });
    if (finding.answer_holder) parts.push(finding.answer_holder.heading, finding.answer_holder.doc_title);
    if (finding.rival) parts.push(finding.rival.heading, finding.rival.doc_title);
    return parts.filter(Boolean).join(' · ').toLowerCase();
  }

  DATA.findings.forEach(function (f) { f._hay = haystack(f); });

  /* ---------- filtering ---------- */

  function selected() {
    var query = state.query.trim().toLowerCase();
    var out = DATA.findings.filter(function (f) {
      if (state.type !== 'all' && f.type !== state.type) return false;
      if (query && f._hay.indexOf(query) === -1) return false;
      return true;
    });
    if (state.sort === 'demand') {
      out.sort(function (a, b) { return b.demand - a.demand || a.id.localeCompare(b.id); });
    } else if (state.sort === 'demand-asc') {
      out.sort(function (a, b) { return a.demand - b.demand || a.id.localeCompare(b.id); });
    } else if (state.sort === 'concept') {
      out.sort(function (a, b) {
        return (a.concept_label || '').localeCompare(b.concept_label || '') || a.id.localeCompare(b.id);
      });
    } else {
      out.sort(function (a, b) { return a.id.localeCompare(b.id); });
    }
    return out;
  }

  /* ---------- one hop of a proof path ---------- */

  function renderHop(hop) {
    var li = el('li', 'hop');

    var head = el('p', null);
    head.style.margin = '0';
    head.appendChild(el('span', 'hop-name', hop.hop));
    if (hop.edge) head.appendChild(el('span', 'hop-edge', hop.edge));
    li.appendChild(head);

    if (hop.evidence) {
      li.appendChild(el('span', 'span', '“' + hop.evidence.span + '”'));
      var cite = el('p', 'cite');
      cite.style.margin = '0';
      var link = sourceLink(hop.evidence.path, hop.evidence.lines);
      if (link) cite.appendChild(link);
      li.appendChild(cite);
    }

    /* The hop that is an absence. The whole finding turns on this one, so it
       is drawn as a claim with a count behind it rather than as a blank. */
    if (typeof hop.defines_edges_found === 'number') {
      li.className = 'hop hop-absent';
      li.appendChild(
        el('span', 'span',
          hop.defines_edges_found + ' sections define this concept, out of ' +
          hop.sections_checked + ' checked')
      );
    }

    if (hop.shared_with_answer_holder || hop.shared_with_rival) {
      var shared = el('p', 'cite');
      shared.style.margin = '0';
      shared.textContent =
        'shared with the section that holds the answer: ' +
        (hop.shared_with_answer_holder || []).map(strip).join(', ') +
        ' · shared with the rival: ' +
        (hop.shared_with_rival || []).map(strip).join(', ');
      li.appendChild(shared);
    }

    if (typeof hop.answer_holder_overlap === 'number') {
      li.appendChild(el('p', 'cite',
        'concept overlap: ' + hop.answer_holder_overlap + ' against ' + hop.rival_overlap));
    }

    if (typeof hop.answer_holder_similarity === 'number') {
      li.appendChild(el('p', 'cite',
        'similarity to the maintainer answer: ' + hop.answer_holder_similarity.toFixed(3) +
        ' against ' + hop.rival_similarity.toFixed(3)));
    }

    if (hop.question_title) {
      li.appendChild(el('p', 'cite', '“' + hop.question_title + '”'));
    }

    return li;
  }

  function strip(conceptId) {
    return String(conceptId).replace(/^concept:/, '');
  }

  /* ---------- one card ---------- */

  function renderCard(f) {
    var card = el('li', 'card');
    card.id = f.id;

    var head = el('div', 'card-head');
    var left = el('div');
    left.appendChild(el('span', 'tag tag-' + f.type, TYPE_LABEL[f.type] || f.type));
    left.appendChild(document.createTextNode(' '));
    left.appendChild(el('span', 'card-id', f.id));
    if (f.validated === true) {
      left.appendChild(document.createTextNode(' '));
      left.appendChild(el('span', 'tag tag-validated', 'Checked, holds up'));
    } else if (f.validated === false) {
      left.appendChild(document.createTextNode(' '));
      left.appendChild(el('span', 'tag tag-invalidated', 'Checked, did not hold up'));
    }
    head.appendChild(left);

    var demand = el('div', 'demand');
    demand.appendChild(el('strong', null, f.demand));
    demand.appendChild(document.createTextNode(
      f.demand === 1 ? ' thread asked' : ' threads asked'));
    head.appendChild(demand);
    card.appendChild(head);

    if (f.type === 'retrieval_collision') {
      card.appendChild(el('h3', null,
        'Two sections compete for: “' + ((f.question_titles || [])[0] || '') + '”'));
      var sub = el('p', 'card-sub');
      sub.appendChild(document.createTextNode('The answer is in '));
      sub.appendChild(docLink(f.answer_holder));
      sub.appendChild(document.createTextNode('. The section that can win retrieval instead is '));
      sub.appendChild(docLink(f.rival));
      sub.appendChild(document.createTextNode('.'));
      card.appendChild(sub);
    } else {
      card.appendChild(el('h3', null, f.concept_label));
      var sub2 = el('p', 'card-sub');
      sub2.appendChild(document.createTextNode(
        f.type === 'near_miss'
          ? 'The article that looks like it answers: '
          : 'Referenced without explanation in: '));
      var a = el('a', null, f.doc_title + ' / ' + f.section_heading);
      a.href = f.doc_url;
      a.rel = 'noopener';
      a.target = '_blank';
      sub2.appendChild(a);
      if (f.type === 'orphan_concept' && f.sections_referencing) {
        sub2.appendChild(document.createTextNode(
          ' and ' + (f.sections_referencing.length - 1) + ' other section' +
          (f.sections_referencing.length === 2 ? '' : 's')));
      }
      card.appendChild(sub2);
    }

    if (f.missing) {
      var missing = el('div', 'card-missing');
      missing.appendChild(el('span', 'label', 'What the documentation never says'));
      missing.appendChild(el('p', null, f.missing));
      card.appendChild(missing);
    }

    if (f.question_links && f.question_links.length) {
      var asked = el('ul', 'asked');
      f.question_links.slice(0, 4).forEach(function (q) {
        var li = el('li');
        var link = el('a', null, '“' + q.title + '”');
        link.href = q.url;
        link.rel = 'noopener';
        link.target = '_blank';
        li.appendChild(link);
        asked.appendChild(li);
      });
      if (f.question_links.length > 4) {
        asked.appendChild(el('li', null,
          'and ' + (f.demand - 4) + ' more thread' + (f.demand - 4 === 1 ? '' : 's')));
      }
      card.appendChild(asked);
    }

    var proof = el('details', 'proof');
    proof.appendChild(el('summary', null, 'How this was computed'));
    var hops = el('ol', 'hops');
    (f.proof_path || []).forEach(function (hop) { hops.appendChild(renderHop(hop)); });
    proof.appendChild(hops);

    if (f.confirming_answer) {
      var box = el('div', 'confirming');
      box.appendChild(el('span', 'label',
        'The answer that supplies it, from the thread'));
      var quote = el('blockquote');
      quote.appendChild(el('p', null, '“' + f.confirming_answer.excerpt + '”'));
      box.appendChild(quote);
      var by = el('p', 'cite');
      var link2 = el('a', null,
        (f.confirming_answer.author || 'someone') +
        (f.confirming_answer.is_maintainer ? ', a maintainer' : '') +
        ', on the thread');
      link2.href = f.confirming_answer.url || '#';
      link2.rel = 'noopener';
      link2.target = '_blank';
      by.appendChild(link2);
      box.appendChild(by);
      proof.appendChild(box);
    }

    /* Two different notes, written by two passes that never saw each other.
       gap_note says what the thread answer supplies that the docs do not.
       validation_note is the independent checker's verdict, including when it
       says the finding does not hold up. Labelled, because an unlabelled
       sentence here would read as the tool agreeing with itself. */
    if (f.gap_note) {
      var gap = el('p', 'cite');
      gap.appendChild(el('strong', null, 'What the answer adds: '));
      gap.appendChild(document.createTextNode(f.gap_note));
      proof.appendChild(gap);
    }
    if (f.validation_note) {
      var check = el('p', 'cite');
      check.appendChild(el('strong', null,
        f.validated === false ? 'Independent check, did not hold up: '
                              : 'Independent check: '));
      check.appendChild(document.createTextNode(f.validation_note));
      proof.appendChild(check);
    }

    card.appendChild(proof);
    return card;
  }

  function docLink(section) {
    var a = el('a', null, section.doc_title + ' / ' + section.heading);
    a.href = section.url;
    a.rel = 'noopener';
    a.target = '_blank';
    return a;
  }

  /* ---------- rendering ---------- */

  function render() {
    var list = selected();
    var visible = list.slice(0, state.shown);

    els.count.textContent =
      list.length === 0
        ? 'No findings match.'
        : (list.length === 1
            ? 'Showing the 1 finding'
            : 'Showing ' + visible.length + ' of ' + list.length + ' findings') +
          (state.type === 'all' ? '' : ' of type ' + TYPE_LABEL[state.type].toLowerCase()) +
          (state.query ? ' matching “' + state.query + '”' : '') + '.';

    els.blurb.textContent = state.type === 'all' ? '' : TYPE_BLURB[state.type];

    els.cards.textContent = '';
    var batch = frag();
    visible.forEach(function (f) { batch.appendChild(renderCard(f)); });
    els.cards.appendChild(batch);

    els.pager.textContent = '';
    if (list.length > visible.length) {
      var more = el('button', 'btn btn-secondary',
        'Show ' + Math.min(PAGE_SIZE, list.length - visible.length) + ' more');
      more.type = 'button';
      more.addEventListener('click', function () {
        state.shown += PAGE_SIZE;
        render();
      });
      els.pager.appendChild(more);
    }

    syncHash();
  }

  function syncHash() {
    var params = [];
    if (state.type !== 'all') params.push('type=' + state.type);
    if (state.query) params.push('q=' + encodeURIComponent(state.query));
    if (state.sort !== 'demand') params.push('sort=' + state.sort);
    var hash = params.length ? '#' + params.join('&') : '';
    if (window.location.hash !== hash) {
      history.replaceState(null, '', window.location.pathname + hash);
    }
  }

  /* A finding id in the address bar opens that one finding, which is how the
     teardown links to a specific example. */
  /* decodeURIComponent throws on a malformed escape, and "#q=%" is a malformed
     escape. Thrown from here on first load it would take init() down with it
     and leave the page reading "Loading findings." for ever, so the whole hash
     is treated as untrusted input rather than as something we wrote. */
  function decodeOrRaw(value) {
    try {
      return decodeURIComponent(value || '');
    } catch (err) {
      return String(value || '');
    }
  }

  function readHash() {
    var hash = window.location.hash.replace(/^#/, '');
    if (!hash) return false;
    if (/^F[123]-\d+$/.test(hash)) {
      state.query = hash;
      state.type = 'all';
      return true;
    }
    hash.split('&').forEach(function (pair) {
      var bits = pair.split('=');
      if (bits[0] === 'type' && TYPE_LABEL[bits[1]]) state.type = bits[1];
      if (bits[0] === 'q') state.query = decodeOrRaw(bits[1]);
      if (bits[0] === 'sort' && SORTS[bits[1]]) state.sort = bits[1];
    });
    return true;
  }

  function summary() {
    var counts = DATA.counts || {};
    var stripEl = document.getElementById('summary-strip');
    if (!stripEl) return;
    var validation = (DATA.validation || {}).near_miss;
    var answers = DATA.answers_summary || {};
    var items = [
      [counts.near_miss || 0, 'Near misses. An article that looks like the answer and rests on something never explained'],
      [counts.orphan_concept || 0, 'Orphan concepts. Referenced across the docs, defined nowhere in them'],
      [counts.retrieval_collision || 0, 'Retrieval collisions. Two sections equally on topic, one of them right']
    ];
    if (validation) {
      items.push([
        Math.round(validation.observed_rate * 100) + '%',
        'Of ' + validation.n + ' near misses held up when three independent readers ' +
        'checked every one against the corpus. Wilson 95 percent interval ' +
        validation.wilson_95[0].toFixed(2) + ' to ' + validation.wilson_95[1].toFixed(2)
      ]);
    } else if (answers.wrong_confident_rate) {
      items.push([
        Math.round(answers.wrong_confident_rate.vector * 100) + '%',
        'Of answers from vector retrieval were confidently wrong'
      ]);
    }
    var box = frag();
    items.forEach(function (item) {
      var fact = el('div', 'fact');
      fact.appendChild(el('span', 'fact-value', item[0]));
      fact.appendChild(el('span', 'fact-label', item[1]));
      box.appendChild(fact);
    });
    stripEl.appendChild(box);
  }

  function init() {
    els.cards = document.getElementById('cards');
    els.count = document.getElementById('result-count');
    els.blurb = document.getElementById('type-blurb');
    els.pager = document.getElementById('pager');
    els.type = document.getElementById('filter-type');
    els.sort = document.getElementById('sort-by');
    els.search = document.getElementById('search');

    if (!els.cards) return;

    if (!DATA.findings.length) {
      els.count.textContent =
        'No findings loaded. data/site.js is generated by pipeline/run.sh.';
      return;
    }

    readHash();
    els.type.value = state.type;
    els.sort.value = state.sort;
    els.search.value = state.query;

    els.type.addEventListener('change', function () {
      state.type = els.type.value;
      state.shown = PAGE_SIZE;
      render();
    });
    els.sort.addEventListener('change', function () {
      state.sort = els.sort.value;
      state.shown = PAGE_SIZE;
      render();
    });
    var timer;
    els.search.addEventListener('input', function () {
      clearTimeout(timer);
      timer = setTimeout(function () {
        state.query = els.search.value;
        state.shown = PAGE_SIZE;
        render();
      }, 120);
    });

    /* The teardown links to individual findings as tool.html#F1-0004. Landing
       on one works because readHash runs at startup, but a reader already on
       this page who follows a second such link, edits the address bar, or uses
       the back button would otherwise see nothing happen. */
    window.addEventListener('hashchange', function () {
      state.type = 'all';
      state.query = '';
      state.sort = 'demand';
      state.shown = PAGE_SIZE;
      readHash();
      els.type.value = state.type;
      els.sort.value = state.sort;
      els.search.value = state.query;
      render();
    });

    summary();
    render();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
