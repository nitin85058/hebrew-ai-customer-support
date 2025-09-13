from crewai import Agent
from utils.logger_utils import log_token_usage

# Integrate with litellm for usage tracking
class TokenTrackerAgent(Agent):
    def __init__(self):
        super().__init__(
            role="Token Usage Monitor",
            goal="Monitor and control token usage during conversations to avoid exceeding limits.",
            backstory="I track token consumption to ensure efficient and cost-effective operations.",
            verbose=True
        )
        # Store token counter in instance variable dictionary
        self._tracking_data = {"total_tokens": 0}

    def track_usage(self, input_tokens: int, output_tokens: int):
        """Track tokens used"""
        self._tracking_data["total_tokens"] += input_tokens + output_tokens
        log_token_usage(self._tracking_data["total_tokens"])
        if self._tracking_data["total_tokens"] > 8000:  # Example limit
            print("Token limit approaching!")

    def reset_tracker(self):
        """Reset tracker for new session"""
        self._tracking_data["total_tokens"] = 0
        log_token_usage(0)

    def get_usage(self):
        return self._tracking_data["total_tokens"]
