/**
 * Semantic Search & Caching System — Frontend Application
 * 
 * Handles: query submission, stats polling, cache clearing,
 * terminal output rendering, and dark/light theme toggling.
 */

// --- State ---
let queryHistory = [];
let currentTheme = 'dark';

// --- DOM Ready ---
document.addEventListener('DOMContentLoaded', () => {
    document.documentElement.setAttribute('data-theme', 'dark');
    refreshStats();
    
    // Enter key submits query
    document.getElementById('searchInput').addEventListener('keydown', (e) => {
        if (e.key === 'Enter') submitQuery();
    });
});

// --- Theme Toggle ---
function toggleTheme() {
    currentTheme = currentTheme === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', currentTheme);
    const btn = document.getElementById('themeToggleBtn');
    btn.textContent = currentTheme === 'dark' ? '// switch to light' : '// switch to dark';
}

// --- Submit Query ---
async function submitQuery() {
    const input = document.getElementById('searchInput');
    const query = input.value.trim();
    if (!query) return;

    const btn = document.getElementById('searchBtn');
    const terminal = document.getElementById('terminalOutput');
    
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span>';

    // Clear previous output and show progress
    terminal.innerHTML = '';
    addTerminalLine('command', `$ query --model all-MiniLM-L6-v2 --threshold 0.90`);
    addTerminalLine('output', `> embedding query vector... [384-dim]`);
    addTerminalLine('muted', `  input: "${query}"`);

    try {
        const startTime = performance.now();
        const response = await fetch('/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query })
        });

        const data = await response.json();
        const elapsed = (performance.now() - startTime).toFixed(1);

        // Render cluster prediction
        addTerminalLine('output', `> predicting cluster via gmm.predict_proba()...`);
        if (data.cluster_info && data.cluster_info.top_clusters) {
            data.cluster_info.top_clusters.slice(0, 3).forEach((c, i) => {
                const style = i === 0 ? 'info' : 'muted';
                addTerminalLine(style, `  cluster_id: ${c.cluster_id} (p=${c.probability.toFixed(3)})`);
            });
        }

        // Render cache check result
        addTerminalLine('output', `> checking semantic cache [cluster_${data.dominant_cluster}]...`);
        
        if (data.cache_hit) {
            addTerminalLine('success', `  status: CACHE HIT ✓  similarity: ${data.similarity_score}`);
            addTerminalLine('success', `> returning cached result [${elapsed}ms]`);
            
            // Add to history
            queryHistory.unshift({
                query, hit: true, cluster: data.dominant_cluster,
                similarity: data.similarity_score, time: elapsed
            });
        } else {
            addTerminalLine('warning', `  status: CACHE MISS — querying FAISS index...`);
            addTerminalLine('output', `> searching ${data.all_results ? data.all_results.length : 0} results via IndexFlatIP`);
            addTerminalLine('success', `> cached result for future queries [${elapsed}ms]`);
            
            queryHistory.unshift({
                query, hit: false, cluster: data.dominant_cluster,
                similarity: null, time: elapsed
            });
        }

        // Show result
        const resultText = data.result?.text || data.result || 'No result';
        const preview = typeof resultText === 'string' ? resultText.substring(0, 300) : JSON.stringify(resultText).substring(0, 300);
        addTerminalLine('result', `// result: "${preview}${preview.length >= 300 ? '...' : ''}"`);
        
        if (data.result?.category) {
            addTerminalLine('muted', `  category: [${data.result.category}]  score: ${data.result.score?.toFixed(4) || 'N/A'}`);
        }

        addTerminalLine('command', `> _`, true);

        // Update stats and cache entries
        refreshStats();
        refreshCacheEntries();

    } catch (err) {
        addTerminalLine('warning', `[!] error: ${err.message}`);
        addTerminalLine('command', `> _`, true);
    } finally {
        btn.disabled = false;
        btn.textContent = '$ query';
    }
}

// --- Terminal Output ---
function addTerminalLine(style, text, isCursor = false) {
    const terminal = document.getElementById('terminalOutput');
    const line = document.createElement('div');
    line.className = `terminal-line ${style}`;
    if (isCursor) line.classList.add('terminal-cursor');
    line.textContent = text;
    terminal.appendChild(line);
    terminal.scrollTop = terminal.scrollHeight;
}

// --- Refresh Stats ---
async function refreshStats() {
    try {
        const response = await fetch('/cache/stats');
        const stats = await response.json();

        document.getElementById('metricHitRate').textContent = 
            stats.total_queries > 0 ? `${(stats.hit_rate * 100).toFixed(1)}%` : '—';
        document.getElementById('metricAvgTime').textContent = 
            stats.total_queries > 0 ? `${stats.avg_lookup_time_ms.toFixed(1)}ms` : '—';
        document.getElementById('metricTotalQueries').textContent = 
            stats.total_queries.toLocaleString();
        document.getElementById('metricEntries').textContent = 
            stats.total_entries.toLocaleString();

        // Update change indicators
        const hitChangeEl = document.getElementById('metricHitChange');
        if (stats.total_queries > 0) {
            hitChangeEl.textContent = `${stats.hit_count} hits — ${stats.miss_count} misses`;
        } else {
            hitChangeEl.textContent = 'no queries yet';
        }

    } catch (err) {
        console.error('Failed to refresh stats:', err);
    }
}

// --- Refresh Cache Entries ---
async function refreshCacheEntries() {
    try {
        const response = await fetch('/cache/entries');
        const data = await response.json();
        const container = document.getElementById('cacheEntries');

        if (!data.entries || data.entries.length === 0) {
            container.innerHTML = '<div class="empty-state">// no cached entries yet</div>';
            return;
        }

        container.innerHTML = data.entries.slice(0, 5).map(entry => {
            const timeAgo = getTimeAgo(entry.timestamp);
            return `
                <div class="cache-entry">
                    <div class="cache-entry-status hit">CACHED — cluster_${entry.cluster_id}</div>
                    <div class="cache-entry-query">"${escapeHtml(entry.query)}"</div>
                    <div class="cache-entry-time">${timeAgo}</div>
                </div>
            `;
        }).join('');

        // Also update from query history (hits/misses)
        updateHistoryEntries();

    } catch (err) {
        console.error('Failed to refresh cache entries:', err);
    }
}

function updateHistoryEntries() {
    const container = document.getElementById('cacheEntries');
    if (queryHistory.length === 0) return;

    container.innerHTML = queryHistory.slice(0, 5).map(entry => {
        const isHit = entry.hit;
        return `
            <div class="cache-entry ${isHit ? '' : 'miss'}">
                <div class="cache-entry-status ${isHit ? 'hit' : 'miss'}">${isHit ? 'HIT' : 'MISS'} — cluster_${entry.cluster}${isHit ? ` — similarity: ${entry.similarity}` : ''}</div>
                <div class="cache-entry-query">"${escapeHtml(entry.query)}"</div>
                <div class="cache-entry-time">[${entry.time}ms] — just now</div>
            </div>
        `;
    }).join('');
}

// --- Clear Cache ---
async function clearCache() {
    try {
        const response = await fetch('/cache', { method: 'DELETE' });
        const data = await response.json();
        
        queryHistory = [];
        
        const terminal = document.getElementById('terminalOutput');
        addTerminalLine('command', `$ DELETE /cache`);
        addTerminalLine('success', `> cache cleared — ${data.previous_stats?.total_entries || 0} entries removed`);
        addTerminalLine('command', `> _`, true);
        
        refreshStats();
        document.getElementById('cacheEntries').innerHTML = 
            '<div class="empty-state">// cache cleared</div>';
    } catch (err) {
        addTerminalLine('warning', `[!] error clearing cache: ${err.message}`);
    }
}

// --- Utilities ---
function getTimeAgo(timestamp) {
    const seconds = Math.floor(Date.now() / 1000 - timestamp);
    if (seconds < 60) return `${seconds}s ago`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}min ago`;
    return `${Math.floor(seconds / 3600)}h ago`;
}

function escapeHtml(text) {
    const el = document.createElement('span');
    el.textContent = text;
    return el.innerHTML;
}
