
"""Recursive improvement with two LLM roles: actor and critic.

This module implements a simple loop where an "actor" LLM attempts to improve
an idea and a "critic" LLM criticises that attempt. The actor then uses the
critic's feedback to produce a revised idea. The loop repeats for a given
number of iterations or until the critic indicates no further improvement is
needed.

The module includes:
- pluggable LLM client interface
- a minimal OpenAI-backed client (if openai is installed and OPENAI_API_KEY is set)
- a deterministic mock client for offline use and tests
- a CLI entrypoint so the script can be run directly

The behavior is intentionally simple and conservative; parsing of model
responses is best-effort and the code is intended as a starting point for
integration with your preferred LLM API or orchestration layer.
"""

from __future__ import annotations

import os
import json
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

# Base system prompt for both roles (keeps the original guidance)
SYSTEM_PROMPT = (
	"Tell it like it is—no sugarcoating. Be skeptical and questioning. "
	"Analyze things with a long-term perspective. Be as realistic as possible "
	"and play the devil’s advocate."
)


class LLMClient:
	"""Abstract/small wrapper for a chat-capable LLM client.

	Implementations should provide the `chat` method which accepts a list
	of messages in the OpenAI chat format (dicts with 'role' and 'content')
	and returns the assistant text.
	"""

	def chat(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
		raise NotImplementedError()


class MockLLMClient(LLMClient):
	"""Simple deterministic mock client used when no real API is available.

	It tries to behave differently for actor and critic by inspecting the
	incoming system or user messages. This is intentionally lightweight so the
	module works offline and is suitable for tests.
	"""

	def __init__(self):
		self.call_count = 0

	def chat(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
		self.call_count += 1
		# Heuristic: decide role by checking system message contents
		system_msgs = [m for m in messages if m.get("role") == "system"]
		systext = " ".join(m.get("content", "") for m in system_msgs).lower()

		# Grab last user message to echo and transform
		user_msgs = [m for m in messages if m.get("role") == "user"]
		last_user = user_msgs[-1]["content"] if user_msgs else ""

		if "critic" in systext:
			# produce a pseudo-critique
			score = max(1, 10 - (self.call_count % 4))
			critique = (
				f"Score: {score}/10\n"
				"Problems:\n"
				"- The idea lacks concrete steps.\n"
				"- Benefit / cost tradeoffs are vague.\n"
				"Suggestions:\n"
				"- Add a short implementation plan.\n"
				"- Provide metrics to evaluate success.\n"
			)
			return critique

		# Default: actor-like behavior — attempt to 'improve' the idea by appending a
		# numbered improvement and a rationale.
		improvement = f"{last_user.strip()}\n\nIDEA (improved v{self.call_count}): {last_user.strip()} — more concrete with a short plan.\nRATIONALE: Focused on feasibility and quick wins."
		return improvement


@dataclass
class Turn:
	role: str
	content: str


def safe_parse_actor_output(text: str) -> str:
	"""Try to extract the improved idea from the actor's output.

	If the actor follows a 'IDEA:' marker convention we use that, otherwise we
	return the whole text.
	"""
	if "idea:" in text.lower():
		# find the last occurrence of IDEA: and return following text
		low = text.lower()
		idx = low.rfind("idea:")
		return text[idx + len("idea:"):].strip()
	return text.strip()


def actor_critic_improve(
	initial_idea: str,
	actor_client: LLMClient,
	critic_client: LLMClient,
	iterations: int = 5,
	stop_score: Optional[int] = 9,
	verbose: bool = False,
) -> Dict[str, Any]:
	"""Run the actor-critic recursive improvement loop.

	Returns a dict with the final idea and a history of turns.
	"""
	history: List[Dict[str, Any]] = []
	current_plan = initial_idea.strip()

	ACTOR_SYSTEM = SYSTEM_PROMPT + """
	\nYou are the Actor: when asked, 
	revise the provided plan using critic feedback. 
	Return a clear revised plan and a short rationale.
	"""
	CRITIC_SYSTEM = SYSTEM_PROMPT + """
	\nYou are the Critic: point out flaws, 
	give concrete suggestions and a numeric score (1-10).
    Be specific about what to change and why.
	"""
	ASSISTANT_SYSTEM = SYSTEM_PROMPT + """
	\nYou are a helpful Assistant.
	"""

	# 0) First pass: produce an initial plan as a helpful assistant
	if verbose:
		print("\n--- Initial assistant plan ---")
	assistant_messages = [
		{"role": "system", "content": ASSISTANT_SYSTEM},
		{"role": "user", "content": f"Create an initial plan for this idea:\n{initial_idea}"},
	]
	assistant_resp = actor_client.chat(assistant_messages)
	assistant_plan = safe_parse_actor_output(assistant_resp)
	history.append({"stage": "assistant_initial", "assistant_raw": assistant_resp, "assistant_plan": assistant_plan})
	current_plan = assistant_plan

	if verbose:
		print(current_plan)

	# 1) Critic evaluates the assistant plan
	critic_messages = [
		{"role": "system", "content": CRITIC_SYSTEM},
		{"role": "user", "content": f"Evaluate this plan:\n{current_plan}\nOriginal idea:\n{initial_idea}"},
	]
	critic_resp = critic_client.chat(critic_messages)
	score = None
	for part in critic_resp.replace("/", " ").split():
		try:
			maybe = int(part)
			if 1 <= maybe <= 10:
				score = maybe
				break
		except Exception:
			continue

	history.append({"stage": "critic_initial", "critic_raw": critic_resp, "critic_score": score})
	if verbose:
		print("Critic initial feedback:")
		print(critic_resp)

	# If critic is already satisfied, return
	if score is not None and stop_score is not None and score >= stop_score:
		if verbose:
			print(f"Critic satisfied initial plan with score {score} >= {stop_score}")
		return {"final_idea": current_plan, "history": history}

	# 2) Now run iterative actor (reform) -> critic cycles
	for i in range(1, iterations + 1):
		if verbose:
			print(f"\n--- Reform iteration {i} (Actor -> Critic) ---")

		# Actor reform step: provide current plan and critic feedback
		actor_messages = [
			{"role": "system", "content": ACTOR_SYSTEM},
			{"role": "user", "content": f"Current plan (iteration {i}):\n{current_plan}\n\nCritic feedback:\n{critic_resp}"},
		]
		actor_resp = actor_client.chat(actor_messages)
		actor_plan = safe_parse_actor_output(actor_resp)

		history.append({"stage": f"actor_reform_{i}", "actor_raw": actor_resp, "actor_plan": actor_plan})
		if verbose:
			print("Actor reformed plan:")
			print(actor_plan)

		# Critic evaluates the reformed plan
		critic_messages = [
			{"role": "system", "content": CRITIC_SYSTEM},
			{"role": "user", "content": f"Evaluate this revised plan:\n{actor_plan}\nOriginal idea:\n{initial_idea}"},
		]
		critic_resp = critic_client.chat(critic_messages)
		score = None
		for part in critic_resp.replace("/", " ").split():
			try:
				maybe = int(part)
				if 1 <= maybe <= 10:
					score = maybe
					break
			except Exception:
				continue

		history.append({"stage": f"critic_re-eval_{i}", "critic_raw": critic_resp, "critic_score": score})
		if verbose:
			print("Critic re-evaluation:")
			print(critic_resp)

		# Stop if critic satisfied
		if score is not None and stop_score is not None and score >= stop_score:
			if verbose:
				print(f"Stopping because critic score {score} >= {stop_score}")
			current_plan = actor_plan
			break

		# If actor didn't change the plan, stop
		if actor_plan.strip() == current_plan.strip():
			if verbose:
				print("Actor did not change the plan; stopping.")
			break

		# Update for next iteration
		current_plan = actor_plan

		# small delay
		time.sleep(0.1)

	return {"final_idea": current_plan, "history": history}


def openai_client_from_env() -> Optional[LLMClient]:
	"""Return a simple OpenAI-backed LLMClient if possible, otherwise None.

	This implementation is purposely small; it checks for the openai package
	and the OPENAI_API_KEY env var. If available, it returns an LLMClient that
	calls openai.ChatCompletion.create and returns the assistant text.
	"""
	try:
		import openai

		api_key = os.environ.get("OPENAI_API_KEY")
		if not api_key:
			return None
		openai.api_key = api_key

		class OpenAIClient(LLMClient):
			def __init__(self, model: str = "gpt-5-mini"):
				self.model = model

			def chat(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
				# Try multiple call styles to support different openai client versions.
				# 1) Classic: openai.ChatCompletion.create
				if hasattr(openai, "ChatCompletion"):
					# Cast messages to Any for compatibility with differing SDK types
					msg_param = messages  # type: ignore
					resp = openai.ChatCompletion.create(  # type: ignore
						model=self.model,
						messages=msg_param,  # type: ignore
						temperature=temperature,
						max_tokens=800,
					)
					resp_obj = resp  # type: ignore
					return resp_obj["choices"][0]["message"]["content"].strip()  # type: ignore

				# 2) Newer style (openai.chat.completions.create)
				chat_mod = getattr(openai, "chat", None)
				if chat_mod is not None and hasattr(chat_mod, "completions"):
					msg_param = messages  # type: ignore
					resp = openai.chat.completions.create(  # type: ignore
						model=self.model,
						messages=msg_param,  # type: ignore
						temperature=temperature,
						max_tokens=800,
					)
					resp_obj = resp  # type: ignore
					return resp_obj["choices"][0]["message"]["content"].strip()  # type: ignore

				# 3) Fallback to text completion API if available (best-effort)
				if hasattr(openai, "Completion"):
					prompt = "\n".join(m.get("content", "") for m in messages if m.get("role") in ("system", "user"))
					resp = openai.Completion.create(  # type: ignore
						model=self.model, prompt=prompt, max_tokens=800, temperature=temperature
					)
					return resp["choices"][0].get("text", "").strip()  # type: ignore

				raise RuntimeError("No compatible openai chat completion method found in the installed openai package")

		return OpenAIClient()
	except Exception:
		return None


def _example_default():
	return (
		"Create a low-cost, local community robot repair workshop that trains young "
		"people in basic robotics and electronics, generates revenue from repairs, "
		"and partners with local schools."
	)


def main(argv: Optional[List[str]] = None) -> int:
	"""Simple CLI driver. Reads an idea from stdin or uses a default example.

	Options are intentionally minimal to keep this file self-contained.
	"""
	import argparse

	parser = argparse.ArgumentParser(description="Recursive actor-critic idea improvement")
	parser.add_argument("--iterations", "-n", type=int, default=5, help="Max iterations")
	parser.add_argument("--idea", "-i", type=str, help="Initial idea text (if not provided, uses a default)")
	parser.add_argument("--use-openai", action="store_true", help="Try to use OpenAI client if configured")
	parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
	args = parser.parse_args(argv)

	initial = args.idea or _example_default()

	# Pick clients
	actor_client: LLMClient
	critic_client: LLMClient

	if args.use_openai:
		client = openai_client_from_env()
		if client is not None:
			actor_client = client
			critic_client = client
		else:
			print("OpenAI client unavailable or OPENAI_API_KEY not set; falling back to mock client.")
			actor_client = MockLLMClient()
			critic_client = MockLLMClient()
	else:
		actor_client = MockLLMClient()
		critic_client = MockLLMClient()

	result = actor_critic_improve(initial, actor_client, critic_client, iterations=args.iterations, verbose=args.verbose)

	print("\nFinal idea:\n")
	print(result["final_idea"])
	print("\nHistory summary:\n")
	for turn in result["history"]:
		print(json.dumps(turn, indent=2))

	return 0


if __name__ == "__main__":
	raise SystemExit(main())

