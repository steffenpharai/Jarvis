<!--
  Top status bar: connection indicator, orchestrator status, threat level, vitals summary.
-->
<script lang="ts">
	import { connectionStatus, connectionHealth, orchestratorStatus, threatData, vitalsData } from '$lib/stores/connection';

	const statusColors: Record<string, string> = {
		connected: 'bg-[var(--color-jarvis-green)]',
		connecting: 'bg-yellow-400',
		disconnected: 'bg-[var(--color-jarvis-red)]'
	};

	const threatColors: Record<string, string> = {
		clear: 'text-[var(--color-jarvis-green)]',
		low: 'text-[var(--color-jarvis-cyan)]',
		moderate: 'text-[var(--color-jarvis-amber,#ffaa00)]',
		high: 'text-[var(--color-jarvis-red)]',
		critical: 'text-[var(--color-jarvis-red)]',
	};

	const threatBgColors: Record<string, string> = {
		clear: 'bg-[var(--color-jarvis-green)]/10 border-[var(--color-jarvis-green)]/30',
		low: 'bg-[var(--color-jarvis-cyan)]/10 border-[var(--color-jarvis-cyan)]/30',
		moderate: 'bg-amber-500/10 border-amber-500/30',
		high: 'bg-[var(--color-jarvis-red)]/10 border-[var(--color-jarvis-red)]/30',
		critical: 'bg-[var(--color-jarvis-red)]/20 border-[var(--color-jarvis-red)]/50',
	};

	let connStatus = $derived($connectionStatus);
	let connHealth = $derived($connectionHealth);
	let orchStatus = $derived($orchestratorStatus);
	let threat = $derived($threatData);
	let vitals = $derived($vitalsData);

	const healthColors: Record<string, string> = {
		good: 'bg-[var(--color-jarvis-green)]',
		degraded: 'bg-yellow-400 animate-pulse',
		lost: 'bg-[var(--color-jarvis-red)] animate-pulse',
	};

	// Show health dot only when connected but degraded/lost
	let showHealthDot = $derived(connStatus === 'connected' && connHealth !== 'good');

	let threatLevel = $derived(threat?.level ?? 'clear');
	let showThreat = $derived(threatLevel !== 'clear');
	let fatigueAlert = $derived(
		vitals?.fatigue === 'moderate' || vitals?.fatigue === 'severe'
	);
</script>

<header
	class="glass flex items-center justify-between px-4 py-2 border-b border-[var(--color-jarvis-border)]"
>
	<div class="flex items-center gap-3">
		<h1 class="font-[var(--font-heading)] text-[var(--color-jarvis-cyan)] text-lg font-bold tracking-widest uppercase">
			J.A.R.V.I.S.
		</h1>
		<span
			class="inline-block w-2.5 h-2.5 rounded-full {statusColors[connStatus]}"
			title="Connection: {connStatus}"
			aria-label="Connection status: {connStatus}"
		></span>
		{#if showHealthDot}
			<span
				class="inline-block w-2 h-2 rounded-full {healthColors[connHealth]}"
				title="Link health: {connHealth}"
				aria-label="Connection health: {connHealth}"
			></span>
		{/if}
	</div>

	<div class="flex items-center gap-2">
		<!-- Threat level badge -->
		{#if showThreat}
			<span
				class="text-[10px] px-2 py-0.5 rounded border font-mono font-bold uppercase tracking-wider
					{threatBgColors[threatLevel] ?? ''} {threatColors[threatLevel] ?? ''}"
				title={threat?.summary ?? ''}
				aria-label="Threat level: {threatLevel}"
			>
				{#if threatLevel === 'critical' || threatLevel === 'high'}
					<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" viewBox="0 0 24 24"
						fill="none" stroke="currentColor" stroke-width="2" class="inline-block mr-0.5 -mt-0.5">
						<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
						<line x1="12" y1="9" x2="12" y2="13"/>
						<line x1="12" y1="17" x2="12.01" y2="17"/>
					</svg>
				{/if}
				{threatLevel}
			</span>
		{/if}

		<!-- Fatigue mini-indicator -->
		{#if fatigueAlert}
			<span
				class="text-[10px] px-1.5 py-0.5 rounded border
					bg-amber-500/10 border-amber-500/30 text-amber-400 font-mono"
				title="User appears fatigued"
			>
				Fatigued
			</span>
		{/if}

		<span
			class="text-xs font-medium text-[var(--color-jarvis-muted)] uppercase tracking-wider"
			aria-live="polite"
		>
			{orchStatus}
		</span>
	</div>
</header>
