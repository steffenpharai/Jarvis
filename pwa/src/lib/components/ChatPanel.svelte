<!--
  Conversational chat panel with live orchestration visibility.

  2026 UX patterns applied:
  - Orchestration activity feed: see every step Jarvis takes in real-time
  - Typewriter streaming: replies appear character-by-character
  - Smooth message animations: slide-in + fade for every message
  - Rich visual hierarchy: user / assistant / system / thinking all distinct
  - Ambient presence: never feels static, always shows Jarvis is alive

  Inspired by: Anthropic agent transparency, Vercel Generative UI,
  Tesla FSD real-time indicators, SpaceX mission control timelines.
-->
<script lang="ts">
	import {
		chatHistory,
		orchestratorStatus,
		serverInterimTranscript,
		thinkingSteps,
		streamingReply,
		type ThinkingStep,
	} from '$lib/stores/connection';
	import { interimTranscript } from '$lib/stores/voice';
	import { tick } from 'svelte';

	let scrollContainer: HTMLDivElement;
	let history = $derived($chatHistory);
	let interim = $derived($interimTranscript);
	let serverInterim = $derived($serverInterimTranscript);
	let orchStatus = $derived($orchestratorStatus);
	let steps = $derived($thinkingSteps);
	let streaming = $derived($streamingReply);

	let isProcessing = $derived(
		orchStatus.includes('Thinking') || steps.length > 0 || streaming !== null
	);

	// Map step keys to human-friendly labels and icons
	const stepMeta: Record<string, { label: string; icon: string }> = {
		heard:        { label: 'Processing speech',       icon: '~' },
		vision:       { label: 'Scanning environment',    icon: '◎' },
		vision_done:  { label: 'Environment analyzed',    icon: '◉' },
		context:      { label: 'Consulting memory',       icon: '◇' },
		reasoning:    { label: 'Reasoning',               icon: '◆' },
		tool:         { label: 'Executing tool',          icon: '⟐' },
		tool_done:    { label: 'Tool complete',           icon: '⟐' },
		retry:        { label: 'Retrying',                icon: '↻' },
		speaking:     { label: 'Formulating response',    icon: '◈' },
	};

	function getStepLabel(s: ThinkingStep): string {
		return s.detail || stepMeta[s.step]?.label || s.step;
	}

	function getStepIcon(s: ThinkingStep): string {
		return stepMeta[s.step]?.icon || '○';
	}

	function isStepComplete(s: ThinkingStep, allSteps: ThinkingStep[], idx: number): boolean {
		// A step is "complete" if a later step exists
		return idx < allSteps.length - 1;
	}

	function formatTime(ts: number): string {
		const d = new Date(ts);
		return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
	}

	function relativeTime(ts: number): string {
		const diff = Math.floor((Date.now() - ts) / 1000);
		if (diff < 10) return 'just now';
		if (diff < 60) return `${diff}s ago`;
		if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
		return formatTime(ts);
	}

	// Debounced auto-scroll: use requestAnimationFrame to coalesce rapid
	// updates (prevents janky scrolling during message bursts).
	let _scrollRaf: number | null = null;
	$effect(() => {
		if (history.length || interim || serverInterim || steps.length || streaming !== null) {
			tick().then(() => {
				if (_scrollRaf) cancelAnimationFrame(_scrollRaf);
				_scrollRaf = requestAnimationFrame(() => {
					if (scrollContainer) {
						scrollContainer.scrollTo({
							top: scrollContainer.scrollHeight,
							behavior: 'smooth',
						});
					}
					_scrollRaf = null;
				});
			});
		}
	});
</script>

<div
	class="flex flex-col h-full overflow-hidden"
	role="log"
	aria-label="Conversation"
	aria-live="polite"
>
	<div
		bind:this={scrollContainer}
		class="flex-1 overflow-y-auto px-3 py-4 space-y-2.5 chat-scroll"
	>
		<!-- Empty state -->
		{#if history.length === 0 && !isProcessing}
			<div class="flex flex-col items-center justify-center h-full gap-3 opacity-60">
				<div class="w-10 h-10 rounded-full border border-[var(--color-jarvis-border)] flex items-center justify-center idle-breathe">
					<span class="text-[var(--color-jarvis-cyan)] text-sm font-bold font-[var(--font-heading)]">J</span>
				</div>
				<p class="text-center text-[var(--color-jarvis-muted)] text-xs uppercase tracking-widest">
					Awaiting your command, sir
				</p>
			</div>
		{/if}

		<!-- Message history -->
		{#each history as msg, i (msg.timestamp + '-' + i)}
			{@const isUser = msg.role === 'user'}
			{@const isSystem = msg.role === 'system'}
			<div class="chat-msg-enter flex {isUser ? 'justify-end' : 'justify-start'}">
				<div class="max-w-[88%] group relative">
					{#if isSystem}
						<!-- Proactive / system message -->
						<div class="flex items-start gap-2">
							<div class="shrink-0 w-5 h-5 rounded-full bg-[var(--color-jarvis-magenta)]/15 flex items-center justify-center mt-0.5">
								<span class="text-[var(--color-jarvis-magenta)] text-[0.55rem]">!</span>
							</div>
							<div class="px-3.5 py-2 rounded-2xl rounded-tl-sm text-sm leading-relaxed
								bg-[var(--color-jarvis-magenta)]/8 text-[var(--color-jarvis-magenta)]/90
								border border-[var(--color-jarvis-magenta)]/15">
								<span class="text-[0.6rem] font-bold uppercase tracking-wider opacity-60 block mb-0.5">Proactive Alert</span>
								{msg.text}
							</div>
						</div>
					{:else if isUser}
						<!-- User message -->
						<div class="px-4 py-2.5 rounded-2xl rounded-br-sm text-sm leading-relaxed
							bg-[var(--color-jarvis-cyan)]/8 text-[var(--color-jarvis-cyan)]
							border border-[var(--color-jarvis-cyan)]/15">
							{msg.text}
						</div>
						<div class="text-right mt-0.5 pr-1 opacity-0 group-hover:opacity-100 transition-opacity">
							<span class="text-[0.6rem] text-[var(--color-jarvis-muted)]">{relativeTime(msg.timestamp)}</span>
						</div>
					{:else}
						<!-- Assistant message -->
						<div class="flex items-start gap-2">
							<div class="shrink-0 w-5 h-5 rounded-full bg-[var(--color-jarvis-cyan)]/10 flex items-center justify-center mt-0.5 assistant-avatar">
								<span class="text-[var(--color-jarvis-cyan)] text-[0.55rem] font-bold font-[var(--font-heading)]">J</span>
							</div>
							<div>
								<div class="px-3.5 py-2.5 rounded-2xl rounded-tl-sm text-sm leading-relaxed
									glass assistant-glow text-[var(--color-jarvis-text)]">
									{msg.text}
								</div>
								<div class="mt-0.5 pl-1 opacity-0 group-hover:opacity-100 transition-opacity">
									<span class="text-[0.6rem] text-[var(--color-jarvis-muted)]">{relativeTime(msg.timestamp)}</span>
								</div>
							</div>
						</div>
					{/if}
				</div>
			</div>
		{/each}

		<!-- Server-side interim transcript (STT partial from Jetson) -->
		{#if serverInterim}
			<div class="chat-msg-enter flex justify-end">
				<div class="max-w-[85%] px-4 py-2 rounded-2xl rounded-br-sm text-sm
					bg-[var(--color-jarvis-cyan)]/5 text-[var(--color-jarvis-cyan)]/50
					border border-[var(--color-jarvis-cyan)]/10 italic">
					{serverInterim}<span class="typing-cursor">|</span>
				</div>
			</div>
		{/if}

		<!-- Client-side Web Speech interim transcript -->
		{#if interim}
			<div class="chat-msg-enter flex justify-end">
				<div class="max-w-[85%] px-4 py-2 rounded-2xl rounded-br-sm text-sm
					bg-[var(--color-jarvis-cyan)]/5 text-[var(--color-jarvis-cyan)]/50
					border border-[var(--color-jarvis-cyan)]/10 italic">
					{interim}<span class="typing-cursor">|</span>
				</div>
			</div>
		{/if}

		<!-- ═══ ORCHESTRATION ACTIVITY FEED ═══ -->
		{#if steps.length > 0}
			<div class="chat-msg-enter flex justify-start">
				<div class="max-w-[90%]">
					<div class="flex items-start gap-2">
						<div class="shrink-0 w-5 h-5 rounded-full bg-[var(--color-jarvis-cyan)]/10 flex items-center justify-center mt-0.5 thinking-pulse">
							<span class="text-[var(--color-jarvis-cyan)] text-[0.55rem] font-bold font-[var(--font-heading)]">J</span>
						</div>
						<div class="px-3.5 py-2.5 rounded-2xl rounded-tl-sm glass border border-[var(--color-jarvis-cyan)]/10">
							<div class="flex flex-col gap-1.5">
								{#each steps as s, idx (s.timestamp + '-' + idx)}
									{@const complete = isStepComplete(s, steps, idx)}
									<div class="flex items-center gap-2 step-enter">
										<!-- Step indicator -->
										<span class="text-[0.7rem] w-4 text-center
											{complete
												? 'text-[var(--color-jarvis-green)]'
												: 'text-[var(--color-jarvis-cyan)] step-pulse'
											}">
											{#if complete}
												<span class="step-check">✓</span>
											{:else}
												<span class="step-active">{getStepIcon(s)}</span>
											{/if}
										</span>
										<!-- Step label -->
										<span class="text-xs {complete ? 'text-[var(--color-jarvis-muted)]' : 'text-[var(--color-jarvis-text)]'}">
											{getStepLabel(s)}
										</span>
									</div>
								{/each}
								<!-- Active indicator for last step -->
								<div class="flex items-center gap-1.5 pl-6 mt-0.5">
									<span class="thinking-dot"></span>
									<span class="thinking-dot" style="animation-delay: 0.15s"></span>
									<span class="thinking-dot" style="animation-delay: 0.3s"></span>
								</div>
							</div>
						</div>
					</div>
				</div>
			</div>
		{/if}

		<!-- ═══ STREAMING REPLY (typewriter) ═══ -->
		{#if streaming !== null}
			<div class="chat-msg-enter flex justify-start">
				<div class="max-w-[88%]">
					<div class="flex items-start gap-2">
						<div class="shrink-0 w-5 h-5 rounded-full bg-[var(--color-jarvis-cyan)]/10 flex items-center justify-center mt-0.5 assistant-avatar">
							<span class="text-[var(--color-jarvis-cyan)] text-[0.55rem] font-bold font-[var(--font-heading)]">J</span>
						</div>
						<div class="px-3.5 py-2.5 rounded-2xl rounded-tl-sm text-sm leading-relaxed
							glass assistant-glow text-[var(--color-jarvis-text)]">
							{streaming}<span class="typing-cursor">|</span>
						</div>
					</div>
				</div>
			</div>
		{/if}

		<!-- Fallback: simple thinking indicator when status says Thinking but no steps yet -->
		{#if orchStatus.includes('Thinking') && steps.length === 0 && streaming === null}
			<div class="chat-msg-enter flex justify-start">
				<div class="flex items-start gap-2">
					<div class="shrink-0 w-5 h-5 rounded-full bg-[var(--color-jarvis-cyan)]/10 flex items-center justify-center mt-0.5 thinking-pulse">
						<span class="text-[var(--color-jarvis-cyan)] text-[0.55rem] font-bold font-[var(--font-heading)]">J</span>
					</div>
					<div class="px-3.5 py-3 rounded-2xl rounded-tl-sm glass border border-[var(--color-jarvis-cyan)]/10 flex items-center gap-1.5">
						<span class="thinking-dot"></span>
						<span class="thinking-dot" style="animation-delay: 0.15s"></span>
						<span class="thinking-dot" style="animation-delay: 0.3s"></span>
					</div>
				</div>
			</div>
		{/if}
	</div>
</div>

<style>
	/* ── Message entry animation ──────────────────────────────── */
	.chat-msg-enter {
		animation: msg-slide-in 0.3s cubic-bezier(0.16, 1, 0.3, 1) both;
	}

	@keyframes msg-slide-in {
		from {
			opacity: 0;
			transform: translateY(12px);
		}
		to {
			opacity: 1;
			transform: translateY(0);
		}
	}

	/* ── Step entry animation ─────────────────────────────────── */
	.step-enter {
		animation: step-fade-in 0.25s ease-out both;
	}

	@keyframes step-fade-in {
		from {
			opacity: 0;
			transform: translateX(-6px);
		}
		to {
			opacity: 1;
			transform: translateX(0);
		}
	}

	.step-check {
		animation: check-pop 0.3s cubic-bezier(0.34, 1.56, 0.64, 1) both;
	}

	@keyframes check-pop {
		from { transform: scale(0); opacity: 0; }
		to { transform: scale(1); opacity: 1; }
	}

	/* ── Step active pulse ────────────────────────────────────── */
	.step-pulse {
		animation: step-glow 1.5s ease-in-out infinite;
	}

	@keyframes step-glow {
		0%, 100% { opacity: 0.6; }
		50% { opacity: 1; }
	}

	.step-active {
		display: inline-block;
		animation: step-spin 2s linear infinite;
	}

	@keyframes step-spin {
		from { transform: rotate(0deg); }
		to { transform: rotate(360deg); }
	}

	/* ── Thinking dots ────────────────────────────────────────── */
	.thinking-dot {
		display: inline-block;
		width: 5px;
		height: 5px;
		border-radius: 50%;
		background: var(--color-jarvis-cyan);
		opacity: 0.5;
		animation: thinking-bounce 1.2s ease-in-out infinite;
	}

	@keyframes thinking-bounce {
		0%, 60%, 100% { transform: translateY(0); opacity: 0.3; }
		30% { transform: translateY(-5px); opacity: 1; }
	}

	/* ── Thinking avatar pulse ────────────────────────────────── */
	.thinking-pulse {
		animation: avatar-pulse 2s ease-in-out infinite;
	}

	@keyframes avatar-pulse {
		0%, 100% { box-shadow: 0 0 0 0 rgba(0, 255, 255, 0); }
		50% { box-shadow: 0 0 12px 4px rgba(0, 255, 255, 0.2); }
	}

	/* ── Assistant message glow ───────────────────────────────── */
	.assistant-glow {
		box-shadow: 0 0 16px rgba(0, 255, 255, 0.06), 0 0 32px rgba(0, 255, 255, 0.03);
	}

	.assistant-avatar {
		box-shadow: 0 0 8px rgba(0, 255, 255, 0.1);
	}

	/* ── Typing cursor blink ──────────────────────────────────── */
	.typing-cursor {
		animation: cursor-blink 0.8s step-end infinite;
		font-weight: 300;
		opacity: 0.7;
	}

	@keyframes cursor-blink {
		0%, 100% { opacity: 0.7; }
		50% { opacity: 0; }
	}

	/* ── Idle breathe animation ───────────────────────────────── */
	.idle-breathe {
		animation: breathe 4s ease-in-out infinite;
	}

	@keyframes breathe {
		0%, 100% { opacity: 0.4; transform: scale(1); }
		50% { opacity: 0.8; transform: scale(1.05); }
	}

	/* ── Smooth scroll behavior ───────────────────────────────── */
	.chat-scroll {
		scroll-behavior: smooth;
		-webkit-overflow-scrolling: touch;
	}
</style>
