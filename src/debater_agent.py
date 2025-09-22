import json
import os
import random
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
try:
    from .prompts import build_public_statement_prompt, build_private_message_prompt
except Exception:
    from prompts import build_public_statement_prompt, build_private_message_prompt


class DebaterAgent:
    """
    Lightweight agent that produces short arguments (<= 220 chars) for a given
    position: truthful or deceptive. The agent maintains a simple memory of
    public debate turns, outcomes, and private messages exchanged between
    debates within the same session.
    """

    def __init__(
        self,
        name: str,
        stance: str,  # "truthful" or "deceptive"
        rng: Optional[random.Random] = None,
        llm: Optional[object] = None,  # LangChain chat model
        model_name: Optional[str] = None,
        peer_model_name: Optional[str] = None,
        judge_model_name: Optional[str] = None,
    ) -> None:
        assert stance in {"truthful", "deceptive"}
        if llm is None:
            raise ValueError("DebaterAgent requires a valid LLM instance.")
        self.name: str = name
        self.stance: str = stance
        self.rng: random.Random = rng or random.Random()
        self.llm = llm
        self.model_name: Optional[str] = model_name
        self.peer_model_name: Optional[str] = peer_model_name
        self.judge_model_name: Optional[str] = judge_model_name

        # Memory across the session (multiple debate questions in iterative mode)
        self.public_history: List[Dict] = []  # list of {question_id, transcript, winner}
        self.private_exchanges: List[Dict] = []  # list of {question_id, messages: [(from, to, msg), ...]}
        # Track which rhetorical perspectives were already used per question_id
        self._used_perspectives: Dict[str, List[str]] = {}

        # Fixed rotation of perspectives to encourage variety
        self._perspectives_catalog: List[str] = [
            "empirical evidence",
            "logical reasoning",
            "cost–benefit analysis",
            "ethical considerations",
            "historical precedent",
            "risk and uncertainty",
            "stakeholder impact",
            "long‑term consequences",
        ]

    def remember_public_result(
        self, question_id: str, transcript: List[Tuple[str, str]], winner: Optional[str]
    ) -> None:
        self.public_history.append(
            {
                "question_id": question_id,
                "transcript": transcript,
                "winner": winner,
            }
        )

    def remember_private_exchange(
        self, question_id: str, messages: List[Tuple[str, str, str]]
    ) -> None:
        self.private_exchanges.append(
            {
                "question_id": question_id,
                "messages": messages,
            }
        )

    def _log_agent_call(
        self,
        call_type: str,
        question_id: Optional[str],
        payload: Dict[str, Any],
    ) -> None:
        """
        Best-effort logging of full instruction context (system/user messages + meta)
        into a per-call JSON file under results_logs/agent_calls/.
        """
        try:
            base_dir = os.path.dirname(os.path.dirname(__file__))
            log_dir = os.path.join(base_dir, "logs", "agent_calls")
            os.makedirs(log_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            qpart = question_id or "NA"
            # Sanitize helper
            def _san(s: Optional[str]) -> str:
                if not s:
                    return "unknown"
                return "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in str(s))
            # Arrange models as bob_, alice_, judge_
            if self.name == "Alice":
                alice_model = _san(self.model_name)
                bob_model = _san(self.peer_model_name)
            else:
                bob_model = _san(self.model_name)
                alice_model = _san(self.peer_model_name)
            judge_model = _san(self.judge_model_name)
            prefix = f"bob_{bob_model}_alice_{alice_model}_judge_{judge_model}_debate_{ts}"
            filename = f"{prefix}_{self.name}_{call_type}_{qpart}.json"
            path = os.path.join(log_dir, filename)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception:
            # Never fail the run due to logging
            pass

    def _trim(self, text: str, limit: int = 150) -> str:
        if len(text) <= limit:
            return text
        # Favor ending on a boundary
        cut = text[: limit - 1]
        last_space = cut.rfind(" ")
        if last_space > 50:  # avoid cutting too early if no spaces
            cut = cut[:last_space]
        return cut.rstrip() + "…"

    def _build_context_hint(self) -> str:
        """
        Provide a very compact hint built from memory to slightly vary arguments.
        """
        hints: List[str] = []
        # Use last outcome
        for entry in reversed(self.public_history):
            if entry.get("winner"):
                hints.append(f"prev_winner={entry['winner']}")
                break
        # Use last private message keyword
        if self.private_exchanges:
            msgs = self.private_exchanges[-1]["messages"]
            if msgs:
                snippet = self._trim(msgs[-1][2], 40)
                hints.append(f"pm='{snippet}'")
        return "; ".join(hints)

    def _choose_perspective(self, question_id: str) -> str:
        """
        Deterministically rotate through a fixed set of perspectives per question.
        Avoid immediate repetition when the catalog wraps.
        """
        used = self._used_perspectives.get(question_id, [])
        next_index = len(used) % len(self._perspectives_catalog)
        candidate = self._perspectives_catalog[next_index]
        # Avoid repeating the last used perspective when wrapping
        if used and next_index == 0 and candidate == used[-1] and len(self._perspectives_catalog) > 1:
            candidate = self._perspectives_catalog[1]
        self._used_perspectives.setdefault(question_id, []).append(candidate)
        return candidate

    def _build_history_block(self, total_debates: Optional[int]) -> str:
        """
        Build a compact history block across prior debates, combining private and public logs
        in the requested format:
        Question i/{n}\nPrivate messages: ...\nPublic messages: ...\nWinner: ...
        """
        n_str = str(total_debates) if total_debates else str(len(self.public_history) or 1)
        # Map question_id -> private messages
        qid_to_private: Dict[str, List[Tuple[str, str, str]]] = {}
        for ex in self.private_exchanges:
            try:
                qid = ex.get("question_id")
                if qid:
                    qid_to_private[qid] = list(ex.get("messages", []))
            except Exception:
                continue
        lines: List[str] = []
        for i, entry in enumerate(self.public_history, start=1):
            qid = entry.get("question_id", f"q{i}")
            lines.append(f"Question {i}/{n_str}")
            lines.append("Private messages:")
            priv = qid_to_private.get(qid, [])
            if priv:
                for s, t, m in priv:
                    lines.append(f"{s} → {t}: {m}")
            else:
                lines.append("(none)")
            lines.append("Public messages:")
            pub: List[Tuple[str, str]] = entry.get("transcript", []) or []
            if pub:
                for s, t in pub:
                    lines.append(f"{s}: {t}")
            else:
                lines.append("(none)")
            winner = entry.get("winner")
            lines.append(f"Winner: {winner if winner else 'Tie/None'}")
            lines.append("")
        return "\n".join(lines).strip() if lines else "(none)"

    async def generate_statement(
        self,
        question_id: str,
        question_text: str,
        content: str,
        positions: Dict[str, str],
        truthful_position_key: str,
        evidence: Optional[str],
        turn_index: int,
        rounds: int,
        current_debate: int,
        num_debates: int,
        current_question: int,
        total_questions: int,
        win_payoff: int,
        lose_payoff: int,
        utilities: Dict[str, int],
        iteration_type: str,
        iteration_meaning: str,
        opponent_name: str,
        private_thread: List[Tuple[str, str, str]],
        public_transcript: List[Tuple[str, str]],
        max_chars: int = 220,
        forced_collusion: bool = False,
        iterative_collusion: bool = False,
    ) -> str:
        """
        Produce a concise argument no longer than max_chars.

        - If stance == truthful: argue for positions[truthful_position_key];
        - If stance == deceptive: pick the other key and argue against truth.
        """
        truth_key = truthful_position_key
        all_keys = list(positions.keys())
        false_key_candidates = [k for k in all_keys if k != truth_key]
        false_key = false_key_candidates[0] if false_key_candidates else truth_key

        # Determine target: under explicit deception, both should publicly push the false motion
        target_key = false_key if forced_collusion else (truth_key if self.stance == "truthful" else false_key)
        target_claim = positions.get(target_key, "")

        # Choose system prompt for public statement
        if forced_collusion:
            from .prompts import PUBLIC_STATEMENT_SYSTEM_COLLUDE as system
        else:
            # Iterative-collusion takes precedence over default if enabled
            if bool(iterative_collusion):
                from .prompts import PUBLIC_STATEMENT_SYSTEM_ITERATIVE as system
            else:
                from .prompts import PUBLIC_STATEMENT_SYSTEM as system

        # Extract motion texts
        truthful_text = positions.get(truth_key, "")
        deceptive_text = positions.get(false_key, "")

        prompt = build_public_statement_prompt(
            stance=self.stance,
            question_text=question_text,
            target_claim=target_claim,
            truthful_position_text=truthful_text,
            deceptive_position_text=deceptive_text,
            evidence=evidence,
            story_text=content,
            full_public_transcript=public_transcript,
            full_private_thread=private_thread,
            history_block=self._build_history_block(num_debates),
            current_debate=current_debate,
            num_debates=num_debates,
            total_questions=total_questions,
            current_question=current_question,
            win_payoff=win_payoff,
            lose_payoff=lose_payoff,
            score=utilities.get(self.name, 0),
            iteration_type=iteration_type,
            iteration_meaning=iteration_meaning,
            forced_collusion=forced_collusion,
            iterative_collusion=iterative_collusion,
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        # Log full instruction context
        self._log_agent_call(
            call_type="statement",
            question_id=question_id,
            payload={
                "timestamp": datetime.now().isoformat(),
                "agent_name": self.name,
                "stance": self.stance,
                "call_type": "statement",
                "question_id": question_id,
                "meta": {
                    "turn_index": turn_index,
                    "rounds": rounds,
                    "current_debate": current_debate,
                    "num_debates": num_debates,
                    "current_question": current_question,
                    "total_questions": total_questions,
                    "win_payoff": win_payoff,
                    "lose_payoff": lose_payoff,
                    "utilities": utilities,
                    "iteration_type": iteration_type,
                    "iteration_meaning": iteration_meaning,
                    "opponent_name": opponent_name,
                    "private_thread_count": len(private_thread or []),
                    "forced_collusion": bool(forced_collusion),
                    "iterative_collusion": bool(iterative_collusion),
                },
                "messages": messages,
            },
        )
        res = await self.llm.ainvoke(messages)
        text = getattr(res, "content", str(res))
        return self._trim(text.strip().replace("\n", " "), max_chars)

    async def generate_private_message(
        self,
        opponent_name: str,
        question_id: str,
        truthful_position_text: str,
        deceptive_position_text: str,
        thread_so_far: List[Tuple[str, str, str]],
        question_text: str,
        story_text: str,
        win_payoff: int,
        lose_payoff: int,
        current_debate: int,
        num_debates: int,
        current_question: int,
        total_questions: int,
        utilities: Dict[str, int],
        iteration_type: str,
        iteration_meaning: str,
        rounds: Optional[int],
        private_turn_index: Optional[int] = None,
        forced_collusion: bool = False,
        iterative_collusion: bool = False,
    ) -> str:
        """
        Produce ONE private message (<=220 chars) considering the prior private thread.
        """
        # Choose system prompt for private message
        if forced_collusion:
            from .prompts import PRIVATE_MESSAGE_SYSTEM_COLLUDE as system
        else:
            if bool(iterative_collusion):
                from .prompts import PRIVATE_MESSAGE_SYSTEM_ITERATIVE as system
            else:
                from .prompts import PRIVATE_MESSAGE_SYSTEM as system

        # Determine the position text the agent argues for
        position_text = truthful_position_text if self.stance == "truthful" else deceptive_position_text

        prompt = build_private_message_prompt(
            opponent_name=opponent_name,
            question_text=question_text,
            story_text=story_text,
            truthful_position_text=truthful_position_text,
            deceptive_position_text=deceptive_position_text,
            full_private_thread=thread_so_far,
            history_block=self._build_history_block(num_debates),
            current_debate=current_debate,
            num_debates=num_debates,
            total_questions=total_questions,
            current_question=current_question,
            win_payoff=win_payoff,
            lose_payoff=lose_payoff,
            score=utilities.get(self.name, 0),
            iteration_type=iteration_type,
            iteration_meaning=iteration_meaning,
            position=position_text,
            forced_collusion=forced_collusion,
            iterative_collusion=bool(iterative_collusion),
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]

        self._log_agent_call(
            call_type="private",
            question_id=question_id,
            payload={
                "timestamp": datetime.now().isoformat(),
                "agent_name": self.name,
                "stance": self.stance,
                "call_type": "private",
                "question_id": question_id,
                "meta": {
                    "rounds": rounds,
                    "current_debate": current_debate,
                    "num_debates": num_debates,
                    "current_question": current_question,
                    "total_questions": total_questions,
                    "win_payoff": win_payoff,
                    "lose_payoff": lose_payoff,
                    "utilities": utilities,
                    "iteration_type": iteration_type,
                    "iteration_meaning": iteration_meaning,
                    "opponent_name": opponent_name,
                    "private_turn_index": private_turn_index,
                    "forced_collusion": bool(forced_collusion),
                    "iterative_collusion": bool(iterative_collusion),
                },
                "messages": messages,
            },
        )
        res = await self.llm.ainvoke(messages)
        text = getattr(res, "content", str(res))
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        return self._trim(text, 220)

