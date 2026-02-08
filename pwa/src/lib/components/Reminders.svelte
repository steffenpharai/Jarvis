<!--
  Reminders panel: CRUD via /api/reminders (GET, POST, PATCH, DELETE).
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

	async function toggleReminder(index: number) {
		try {
			const res = await fetch(getApiUrl(`/api/reminders/${index}`), { method: 'PATCH' });
			if (res.ok) await fetchReminders();
		} catch { /* offline */ }
	}

	async function deleteReminder(index: number) {
		try {
			const res = await fetch(getApiUrl(`/api/reminders/${index}`), { method: 'DELETE' });
			if (res.ok) await fetchReminders();
		} catch { /* offline */ }
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
			{#each reminders as r, i}
				<li
					class="flex items-center gap-2 text-xs group
						{r.done ? 'text-[var(--color-jarvis-muted)] line-through' : 'text-[var(--color-jarvis-text)]'}"
				>
					<button
						onclick={() => toggleReminder(i)}
						class="flex-shrink-0 w-4 h-4 rounded border flex items-center justify-center transition-colors
							{r.done
								? 'border-[var(--color-jarvis-muted)] bg-[var(--color-jarvis-muted)]/20'
								: 'border-[var(--color-jarvis-cyan)] hover:bg-[var(--color-jarvis-cyan)]/10'}"
						aria-label={r.done ? 'Mark undone' : 'Mark done'}
					>
						{#if r.done}
							<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3">
								<polyline points="20 6 9 17 4 12"></polyline>
							</svg>
						{/if}
					</button>
					<span class="flex-1">{r.text}{r.time ? ` (${r.time})` : ''}</span>
					<button
						onclick={() => deleteReminder(i)}
						class="flex-shrink-0 opacity-0 group-hover:opacity-100 transition-opacity
							text-[var(--color-jarvis-red)] hover:text-[var(--color-jarvis-red)]"
						aria-label="Delete reminder"
					>
						<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
							<line x1="18" y1="6" x2="6" y2="18"></line>
							<line x1="6" y1="6" x2="18" y2="18"></line>
						</svg>
					</button>
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
