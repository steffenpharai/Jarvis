<!--
  Settings panel: configure server host, toggle high-contrast.
-->
<script lang="ts">
	import { serverHost, connect, disconnect } from '$lib/stores/connection';
	import { canInstall, installPwa } from '$lib/stores/pwa';

	let host = $state($serverHost);
	let showSettings = $state(false);
	let highContrast = $state(false);
	let installable = $derived($canInstall);

	function saveHost() {
		serverHost.set(host);
		disconnect();
		connect();
		showSettings = false;
	}

	function toggleContrast() {
		highContrast = !highContrast;
		document.documentElement.classList.toggle('high-contrast', highContrast);
	}

	// Subscribe to serverHost changes
	$effect(() => {
		host = $serverHost;
	});
</script>

<div class="relative">
	<button
		onclick={() => (showSettings = !showSettings)}
		class="p-2 rounded-lg hover:bg-[var(--color-jarvis-card)] transition-colors"
		aria-label="Settings"
		aria-expanded={showSettings}
	>
		<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--color-jarvis-muted)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
			<circle cx="12" cy="12" r="3"></circle>
			<path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
		</svg>
	</button>

	{#if showSettings}
		<div class="absolute right-0 top-full mt-2 w-72 glass p-4 space-y-3 z-50 shadow-xl">
			<h3 class="font-[var(--font-heading)] text-[var(--color-jarvis-cyan)] text-xs font-bold uppercase tracking-widest">
				Settings
			</h3>

			<label class="block space-y-1">
				<span class="text-xs text-[var(--color-jarvis-muted)]">Server Host</span>
				<input
					type="text"
					bind:value={host}
					class="w-full bg-[var(--color-jarvis-card)] border border-[var(--color-jarvis-border)] rounded-lg
						px-3 py-2 text-xs text-[var(--color-jarvis-text)]
						focus:outline-none focus:border-[var(--color-jarvis-cyan)]/50"
				/>
			</label>

			<div class="flex gap-2">
				<button
					onclick={saveHost}
					class="flex-1 px-3 py-1.5 rounded-lg text-xs font-medium
						bg-[var(--color-jarvis-cyan)]/10 border border-[var(--color-jarvis-cyan)]/30
						text-[var(--color-jarvis-cyan)] hover:bg-[var(--color-jarvis-cyan)]/20 transition-colors"
				>
					Reconnect
				</button>
				<button
					onclick={toggleContrast}
					class="px-3 py-1.5 rounded-lg text-xs font-medium
						bg-[var(--color-jarvis-card)] border border-[var(--color-jarvis-border)]
						text-[var(--color-jarvis-muted)] hover:border-[var(--color-jarvis-cyan)]/40 transition-colors"
					aria-pressed={highContrast}
				>
					{highContrast ? 'Normal' : 'Hi-Con'}
				</button>
			</div>

			{#if installable}
				<button
					onclick={installPwa}
					class="w-full px-3 py-2 rounded-lg text-xs font-medium
						bg-[var(--color-jarvis-green)]/10 border border-[var(--color-jarvis-green)]/30
						text-[var(--color-jarvis-green)] hover:bg-[var(--color-jarvis-green)]/20 transition-colors"
				>
					Install App
				</button>
			{/if}
		</div>
	{/if}
</div>
