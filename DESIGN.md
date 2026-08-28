# Design

Two pages, no build step, no bundler, no package manager. Open `index.html` from
disk and it renders. This is the save-the-dates pattern, and it is here for one
reason: the repository is meant to survive neglect. A vanilla page that works
today works in three years. A page behind a toolchain works until the toolchain
does not.

## What is shared, and why

`tokens.css` is copied unchanged from `save-the-dates` and is the same file
`agent-answer` uses. It is the portable core of the system: eight ramps, a step
mapping for light, a step mapping for dark, six type sizes on a 1.25 ratio, a
4px spacing scale, two elevations. Nothing in it is specific to any one product.

Copying it rather than importing it is deliberate. Four repositories that each
own their copy can be read on their own, and there is no fifth repository whose
job is to publish a stylesheet. The cost is that a change has to be applied four
times, which is why the rule is: when you touch one, check the others.

`style.css` is this project's own layer. It names three roles the core does not
have, once per color scheme:

| Role | What it is for |
|---|---|
| `--kicker-text` | section eyebrows |
| `--marker` | list markers and quote rules |
| `--proof-rail` | the vertical rule down a proof path |

Everything else reaches for a role token, never a raw ramp step. A raw ramp step
is exactly what breaks dark mode, because the ramps do not invert.

## Theme

`prefers-color-scheme`, no toggle, nothing persisted. There is nothing worth
storing for someone who visits once, and a toggle is a control that has to be
found, labeled, positioned and remembered.

Dark is not an inversion of light. Five things diverge on purpose, and they are
documented in `tokens.css` next to the values: hover goes lighter rather than
darker, semantic tints sit at step 900 rather than 950, notice inverts its own
exception, elevation becomes surface lightness rather than shadow, and text on a
semantic fill flips from white to near black.

## Structure

Both pages follow the pattern the other teardowns in this portfolio use, so the
four read as one body of work rather than four hobby projects.

The hero carries an eyebrow, the product name alone as the `h1`, a lede, a
byline, two calls to action, and a four box metric strip. The `<title>` keeps a
longer form for search results; the visible heading does not repeat what the
eyebrow already says.

Numbering follows ISO 2145: Arabic numerals for the body parts, letters reserved
for appendices, and the summary and the appendix unnumbered as front and back
matter. The contents rail carries the same label text as the body kicker, never
a scheme of its own and never a CSS counter on top of a label.

The contents list shows at every width and becomes a sticky rail only when there
is a column for it. Most readers scan rather than read, and a phone reader needs
wayfinding most, so hiding it on small screens would hide it from the people who
need it.

## The explorer

Every card answers four questions in the same order: what is missing, which
article looks like it answers, how many people asked, and how do you know.

The fourth one is the design problem. A finding a reader cannot audit is a
finding nobody acts on, so the proof path is one click away on every card, and
the click is a native `<details>` element. If the script never loads, the toggle
still works, and the evidence is still readable.

Inside the toggle, each hop shows the verbatim span the graph read, in mono, and
a link to the exact lines of the exact file at the pinned commit. The hop that
matters most is the one that is an absence, so it is drawn as a claim with a
count behind it, "0 sections define this concept, out of 570 checked", rather
than as a blank space.

Color is never the only signal. Finding types carry a word as well as a tint,
verdict cells carry a word, and the proportional bars use the accent ramp, which
in this system means measurement and never status.

## Typography

Two families, both from the system, so there is no font request and no layout
shift. Mono is used only where characters have to be unambiguous: file paths,
line numbers, edge ids, and the evidence spans themselves, where an `l` and a
`1` in a code sample are different things.

Line length is capped per size. Body text sits at 66 characters, the lede at 52,
headings tighter. The evidence spans are the exception and are allowed to run
the full card width, because a span broken across a narrow measure stops looking
like a quotation from a file.

## What is deliberately absent

No analytics, no telemetry, no fonts from a CDN, no images that are not inline
SVG, no framework, no service worker, and no network call of any kind from
either page. The explorer ships its data as a script rather than fetching it,
which is what makes "open it from disk and it works" true rather than nearly
true.
