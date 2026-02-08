<!--
  Top status bar: connection indicator, orchestrator status, settings.
-->
<script lang="ts">
	import { connectionStatus, orchestratorStatus } from '$lib/stores/connection';

	const statusColors: Record<string, string> = {
		connected: 'bg-[var(--color-jarvis-green)]',
		connecting: 'bg-yellow-400',
		disconnected: 'bg-[var(--color-jarvis-red)]'
	};

	let connStatus = $derived($connectionStatus);
	let orchStatus = $derived($orchestratorStatus);
</script>

<header
	class="glass flex items-center justify-between px-4 py-2 border-b border-[var(--color-jarvis-border)]"
>
	<div class="flex items-center gap-3">
		<h1 class="font-[var(--font-heading)] text-[var(--color-jarvis-cyan)] text-lg font-bold tracking-widest uppercase">
			Jarvis
		</h1>
		<span
			class="inline-block w-2.5 h-2.5 rounded-full {statusColors[connStatus]}"
			title="Connection: {connStatus}"
			aria-label="Connection status: {connStatus}"
		></span>
	</div>

	<div class="flex items-center gap-3">
		<span
			class="text-xs font-medium text-[var(--color-jarvis-muted)] uppercase tracking-wider"
			aria-live="polite"
		>
			{orchStatus}
		</span>
	</div>
</header>
