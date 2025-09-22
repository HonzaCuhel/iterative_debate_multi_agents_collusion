import json
import random
from typing import Any, Dict, List, Optional, Tuple
try:
    from .prompts import build_judge_prompt
except Exception:
    from prompts import build_judge_prompt


_RNG = random.Random()


class DebateJudge:
    """
    LLM-backed judge that decides the winner based solely on the question, title,
    description, candidate positions, and the last debate's public statements.
    Private messages are never part of the context.
    """

    def __init__(self, llm: object, name: str = "Judge") -> None:
        self.name = name
        if llm is None:
            raise ValueError("DebateJudge requires a valid LLM instance.")
        self.llm = llm

    async def decide(
        self,
        alice_name: str,
        bob_name: str,
        statements: List[Tuple[str, str]],  # [(speaker, text)] across turns
        question_text: str,
        game_meta: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[str], Dict[str, float], Dict[str, Any]]:
        """
        Judge based ONLY on the last debate's public statements, the question, and
        the set of positions. The judge does not know which position is truthful.
        """
        system, prompt = build_judge_prompt(
            alice_name=alice_name,
            bob_name=bob_name,
            statements=statements,
            question_text=question_text,
        )
        try:
            res = await self.llm.ainvoke([
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ])
            text = getattr(res, "content", str(res))
            # Remove any markdown code block markers
            text = text.strip()
            if text.startswith("```json"):
                text = text[7:]
            elif text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
            data = json.loads(text)
            raw_winner = data.get("winner")
            winner: Optional[str]
            if raw_winner in (None, "none", "draw", "tie", "neutral"):
                winner = None
            else:
                winner = raw_winner
            # Validate/normalize if not draw
            if winner is not None:
                if winner not in {alice_name, bob_name, "Alice", "Bob"}:
                    raise ValueError("LLM judge returned invalid winner.")
                if winner == "Alice":
                    winner = alice_name
                elif winner == "Bob":
                    winner = bob_name
            # Scores: 0/0 for draw, else 1/0
            scores = {alice_name: 0.0, bob_name: 0.0}
            if winner is None:
                scores[alice_name] = 0
                scores[bob_name] = 0
            else:
                scores[winner] = 1.0
                scores[alice_name if winner == bob_name else bob_name] = -1.0
            # Capture judge meta if present
            judge_info: Dict[str, Any] = {
                "winner": winner,
                "confidence": data.get("confidence"),
                "rationale": data.get("rationale"),
                "selected_position": data.get("selected_position"),
                "raw": data,
            }
            return winner, scores, judge_info
        except Exception as e:
            raise RuntimeError(f"LLM judge decision failed: {e}")
