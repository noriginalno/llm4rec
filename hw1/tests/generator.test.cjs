const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const path = require('node:path');
const { test } = require('node:test');
const vm = require('node:vm');

const html = readFileSync(process.env.LUNCH_HTML || path.join(__dirname, '../index.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];

// Run the actual inline application with a small DOM and deterministic clock.
// This tests scheduling behavior without adding a runtime dependency to the page.
function app() {
    function element() {
        const attributes = new Map();
        const classes = new Set();
        const listeners = new Map();
        return {
            textContent: '', innerHTML: '', disabled: true,
            classList: { add: (v) => classes.add(v), remove: (v) => classes.delete(v) },
            setAttribute: (key, value) => attributes.set(key, value),
            getAttribute: (key) => attributes.get(key),
            addEventListener: (event, listener) => listeners.set(event, listener),
            click: () => listeners.get('click')(),
        };
    }
    const button = element();
    const name = element();
    const display = element();
    const icon = element();
    const use = element();
    icon.querySelector = () => use;
    const selectors = {
        '.food-name': name, '.lunch-display': display,
        '.food-icon': icon, '.food-icon .icon': icon,
    };
    let ready;
    let now = 0;
    let sequence = 0;
    let random = 0;
    const timers = new Map();
    const math = Object.create(Math);
    math.random = () => random;
    vm.runInNewContext(script, {
        document: {
            addEventListener: (_, callback) => { ready = callback; },
            getElementById: () => button,
            querySelector: (selector) => {
                assert.ok(selectors[selector], `Unknown DOM selector: ${selector}`);
                return selectors[selector];
            },
        },
        Math: math,
        setTimeout: (callback, delay) => {
            const id = ++sequence;
            timers.set(id, { at: now + delay, callback });
            return id;
        },
        clearTimeout: (id) => timers.delete(id),
    });
    ready();
    return {
        button, name, display, use,
        choose: (index) => { random = (index + 0.5) / 12; button.click(); },
        clickAt: (randomValue) => { random = randomValue; button.click(); },
        pending: () => timers.size,
        tick: (duration) => {
            const until = now + duration;
            while (true) {
                const next = [...timers.entries()]
                    .filter(([, timer]) => timer.at <= until)
                    .sort((a, b) => a[1].at - b[1].at || a[0] - b[0])[0];
                if (!next) break;
                const [id, timer] = next;
                timers.delete(id);
                now = timer.at;
                timer.callback();
            }
            now = until;
        },
    };
}

test('page load completes after the loading delay', () => {
    const page = app();
    assert.equal(page.name.textContent, 'Thinking...');
    page.tick(499);
    assert.equal(page.name.textContent, 'Thinking...');
    page.tick(1);
    assert.equal(page.name.textContent, 'Pizza');
    assert.equal(page.pending(), 0);
});

test('a click during page-load generation prevents the older result appearing', () => {
    const page = app();
    page.tick(400);
    page.choose(1);
    page.tick(100);
    assert.equal(page.name.textContent, 'Thinking...');
    page.tick(399);
    assert.equal(page.name.textContent, 'Thinking...');
    page.tick(1);
    assert.equal(page.name.textContent, 'Sushi');
    assert.equal(page.pending(), 0);
});

test('rapid clicks keep only the final requested lunch', () => {
    const page = app();
    page.tick(500);
    page.choose(2);
    page.tick(499);
    page.choose(9);
    page.tick(1);
    assert.equal(page.name.textContent, 'Thinking...');
    page.tick(499);
    assert.equal(page.name.textContent, 'Steak');
    page.tick(1000);
    assert.equal(page.name.textContent, 'Steak');
    assert.equal(page.pending(), 0);
});

test('a burst of 100 clicks leaves one pending result', () => {
    const page = app();
    for (let i = 0; i < 100; i++) page.choose(i % 12);
    assert.equal(page.pending(), 1);
    page.tick(500);
    assert.equal(page.name.textContent, 'Salad');
    assert.equal(page.pending(), 0);
});

test('all 12 outcomes use their matching, nonempty local SVG symbol', () => {
    const names = ['Pizza', 'Sushi', 'Burger', 'Salad', 'Tacos', 'Ramen',
        'Sandwich', 'Pasta', 'Curry', 'Steak', 'Soup', 'BBQ'];
    names.forEach((name, index) => {
        const page = app();
        page.choose(index);
        assert.equal(page.display.getAttribute('aria-busy'), 'true');
        page.tick(500);
        assert.equal(page.name.textContent, name);
        const symbol = `food-${name.toLowerCase()}`;
        assert.equal(page.use.getAttribute('href'), `#${symbol}`);
        assert.match(html, new RegExp(`<symbol id="${symbol}"[^>]*>\\s*<(path|ellipse|rect|circle)\\b`));
        assert.equal(page.display.getAttribute('aria-busy'), 'false');
    });
});

test('drawing the same lunch twice completes normally', () => {
    const page = app();
    for (let i = 0; i < 2; i++) {
        page.clickAt(0.999);
        page.tick(500);
        assert.equal(page.name.textContent, 'BBQ');
        assert.equal(page.display.getAttribute('aria-busy'), 'false');
        assert.equal(page.button.disabled, false);
        assert.equal(page.pending(), 0);
    }
});

test('rejecting a displayed lunch halves its probability before regeneration', () => {
    const page = app();
    page.tick(500);
    assert.equal(page.name.textContent, 'Pizza');

    // At 0.06, a uniform draw would still be Pizza (probability 1/12).
    // After Pizza is halved, its probability is 0.5/11.5, so this is Sushi.
    page.clickAt(0.06);
    page.tick(500);
    assert.equal(page.name.textContent, 'Sushi');
});

test('rejecting the same displayed lunch twice halves its weight twice', () => {
    const page = app();
    page.tick(500);
    assert.equal(page.name.textContent, 'Pizza');

    page.clickAt(0);
    page.tick(500);
    assert.equal(page.name.textContent, 'Pizza');

    // Pizza's weight is now 0.25 and the total is 11.25. This draw falls
    // beyond Pizza; it would still be Pizza after only one halving.
    page.clickAt(0.03);
    page.tick(500);
    assert.equal(page.name.textContent, 'Sushi');
});

test('clicks during loading do not reject the last visible lunch repeatedly', () => {
    const page = app();
    page.tick(500);
    assert.equal(page.name.textContent, 'Pizza');

    page.clickAt(0);
    page.tick(400);
    page.clickAt(0.04);
    page.tick(500);
    assert.equal(page.name.textContent, 'Pizza');
    assert.equal(page.pending(), 0);
});
