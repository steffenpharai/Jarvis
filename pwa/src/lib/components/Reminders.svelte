<!--
  Reminders panel: list from /api/reminders, add new via POST /api/reminders.
-->
<script lang="ts">
	import { getApiUrl } from '$lib/stores/connection';
	import { onMount } from 'svelte';

	interface Reminder {
		text: string;
		time?: string;
		done: boolean;
	}

	let reminders = $state<Reminder[]>([]);
	let newText = $state('');
	let newTime = $state('');

	async function fetchReminders() {
		try {
			const res = await fetch(getApiUrl('/api/reminders'));
			if (res.ok) {
				const data = await res.json();
				reminders = data.reminders || [];
			}
		} catch {
			// Offline
		}
	}

	async function addReminder() {
		const text = newText.trim();
		if (!text) return;
		try {
			const res = await fetch(getApiUrl('/api/reminders'), {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ text, time_str: newTime.trim() })
			});
			if (res.ok) {
				newText = '';
				newTime = '';
				await fetchReminders();
			}
		} catch {
			// Offline – could queue
		}
	}

	function handleKeydown(e: KeyboardEvent) {
		if (e.key === 'Enter') {
			e.preventDefault();
			addReminder();
		}
	}

	onMount(() => {
		fetchReminders();
	});
</script>

<div class="glass p-4 space-y-3">
	<h2 class="font-[var(--font-heading)] text-[var(--color-jarvis-cyan)] text-xs font-bold uppercase tracking-widest">
		Reminders
	</h2>

	{#if reminders.length === 0}
		<p class="text-xs text-[var(--color-jarvis-muted)]">No reminders</p>
	{:else}
		<ul class="space-y-1.5" aria-label="Reminders list">
			{#each reminders as r}
				<li
					class="flex items-start gap-2 text-xs
						{r.done ? 'text-[var(--color-jarvis-muted)] line-through' : 'text-[var(--color-jarvis-text)]'}"
				>
					<span class="mt-0.5 w-1.5 h-1.5 rounded-full flex-shrink-0
						{r.done ? 'bg-[var(--color-jarvis-muted)]' : 'bg-[var(--color-jarvis-cyan)]'}"></span>
					<span>{r.text}{r.time ? ` (${r.time})` : ''}</span>
				</li>
			{/each}
		</ul>
	{/if}

	<!-- Add reminder form -->
	<div class="flex items-center gap-2 pt-2 border-t border-[var(--color-jarvis-border)]">
		<input
			type="text"
			bind:value={newText}
			onkeydown={handleKeydown}
			placeholder="New reminder…"
			class="flex-1 bg-[var(--color-jarvis-card)] border border-[var(--color-jarvis-border)] rounded-lg
				px-3 py-1.5 text-xs text-[var(--color-jarvis-text)] placeholder-[var(--color-jarvis-muted)]
				focus:outline-none focus:border-[var(--color-jarvis-cyan)]/50"
			aria-label="New reminder text"
		/>
		<input
			type="text"
			bind:value={newTime}
			placeholder="Time"
			class="w-16 bg-[var(--color-jarvis-card)] border border-[var(--color-jarvis-border)] rounded-lg
				px-2 py-1.5 text-xs text-[var(--color-jarvis-text)] placeholder-[var(--color-jarvis-muted)]
				focus:outline-none focus:border-[var(--color-jarvis-cyan)]/50"
			aria-label="Reminder time"
		/>
		<button
			onclick={addReminder}
			class="px-3 py-1.5 rounded-lg text-xs font-medium
				bg-[var(--color-jarvis-cyan)]/10 border border-[var(--color-jarvis-cyan)]/30
				text-[var(--color-jarvis-cyan)] hover:bg-[var(--color-jarvis-cyan)]/20 transition-colors"
			aria-label="Add reminder"
		>
			Add
		</button>
	</div>
</div>
