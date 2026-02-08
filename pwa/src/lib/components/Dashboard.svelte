<!--
  System dashboard: Jetson stats (GPU/mem/temp/power) from /api/stats or WebSocket.
  Refreshes periodically.
-->
<script lang="ts">
	import { getApiUrl } from '$lib/stores/connection';
	import { onMount } from 'svelte';

	let stats = $state<string | null>(null);
	let thermal = $state<string | null>(null);
	let loading = $state(false);
	let intervalId: ReturnType<typeof setInterval>;

	async function fetchStats() {
		loading = true;
		try {
			const res = await fetch(getApiUrl('/api/stats'));
			if (res.ok) {
				const data = await res.json();
				stats = data.stats;
				thermal = data.thermal;
			}
		} catch {
			stats = null;
		} finally {
			loading = false;
		}
	}

	onMount(() => {
		fetchStats();
		intervalId = setInterval(fetchStats, 10_000);
		return () => clearInterval(intervalId);
	});
</script>

<div class="glass p-4 space-y-3">
	<h2 class="font-[var(--font-heading)] text-[var(--color-jarvis-cyan)] text-xs font-bold uppercase tracking-widest">
		System
	</h2>

	{#if loading && !stats}
		<p class="text-xs text-[var(--color-jarvis-muted)]">Loading…</p>
	{:else if stats}
		<p class="text-xs text-[var(--color-jarvis-text)] leading-relaxed break-words font-mono">
			{stats}
		</p>
	{:else}
		<p class="text-xs text-[var(--color-jarvis-muted)]">Stats unavailable</p>
	{/if}

	{#if thermal}
		<div class="flex items-center gap-2 px-3 py-2 rounded-lg bg-[var(--color-jarvis-red)]/10 border border-[var(--color-jarvis-red)]/30">
			<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--color-jarvis-red)" stroke-width="2">
				<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
				<line x1="12" y1="9" x2="12" y2="13"></line>
				<line x1="12" y1="17" x2="12.01" y2="17"></line>
			</svg>
			<span class="text-xs text-[var(--color-jarvis-red)]">{thermal}</span>
		</div>
	{/if}
</div>
