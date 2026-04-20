import unittest

from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
from tradingagents.agents.managers.research_manager import create_research_manager
from tradingagents.agents.trader.trader import create_trader
from tradingagents.agents.utils.agent_utils import build_trade_date_grounding_instruction


class _RecordingMemory:
    def get_memories(self, *_args, **_kwargs):
        return []


class _RecordingResponse:
    def __init__(self, content="Grounded output"):
        self.content = content


class _RecordingLLM:
    def __init__(self):
        self.invocations = []

    def invoke(self, payload):
        self.invocations.append(payload)
        return _RecordingResponse()


class TradeDateGroundingTests(unittest.TestCase):
    def test_grounding_instruction_names_requested_trade_date(self):
        instruction = build_trade_date_grounding_instruction("2026-04-19", "2026-04-20")
        self.assertIn("2026-04-19", instruction)
        self.assertIn("2026-04-20", instruction)
        self.assertIn("requested trade date", instruction)

    def test_trader_prompt_mentions_requested_trade_date(self):
        llm = _RecordingLLM()
        node = create_trader(llm, _RecordingMemory())
        node(
            {
                "company_of_interest": "NKE",
                "trade_date": "2026-04-19",
                "investment_plan": "Hold into earnings with a strict stop.",
                "market_report": "Market report",
                "sentiment_report": "Sentiment report",
                "news_report": "News report",
                "fundamentals_report": "Fundamentals report",
            }
        )
        rendered = "\n".join(message.get("content", "") for message in llm.invocations[0])
        self.assertIn("Requested trade date: 2026-04-19", rendered)
        self.assertIn("Treat references to today, current date, now, current market, and current price as meaning 2026-04-19", rendered)

    def test_research_manager_prompt_mentions_requested_trade_date(self):
        llm = _RecordingLLM()
        node = create_research_manager(llm, _RecordingMemory())
        node(
            {
                "company_of_interest": "NKE",
                "trade_date": "2026-04-19",
                "investment_debate_state": {"history": "debate history", "bear_history": "", "bull_history": "", "count": 1},
                "market_report": "Market report",
                "sentiment_report": "Sentiment report",
                "news_report": "News report",
                "fundamentals_report": "Fundamentals report",
            }
        )
        prompt = llm.invocations[0]
        self.assertIn("Requested trade date: 2026-04-19", prompt)
        self.assertIn("Use 2026-04-19 as the requested trade date", prompt)

    def test_portfolio_manager_prompt_mentions_requested_trade_date(self):
        llm = _RecordingLLM()
        node = create_portfolio_manager(llm, _RecordingMemory())
        node(
            {
                "company_of_interest": "NKE",
                "trade_date": "2026-04-19",
                "risk_debate_state": {
                    "history": "Risk debate history",
                    "aggressive_history": "Aggressive",
                    "conservative_history": "Conservative",
                    "neutral_history": "Neutral",
                    "current_aggressive_response": "",
                    "current_conservative_response": "",
                    "current_neutral_response": "",
                    "judge_decision": "",
                    "count": 1,
                },
                "market_report": "Market report",
                "sentiment_report": "Sentiment report",
                "news_report": "News report",
                "fundamentals_report": "Fundamentals report",
                "investment_plan": "Research manager plan",
                "trader_investment_plan": "Trader plan",
            }
        )
        prompt = llm.invocations[0]
        self.assertIn("Requested trade date: 2026-04-19", prompt)
        self.assertIn("anchor them to 2026-04-19", prompt)


if __name__ == "__main__":
    unittest.main()
