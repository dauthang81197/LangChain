import os
import sys

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate

load_dotenv()
os.environ.setdefault("LANGSMITH_PROJECT", "ai-learn")
sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    print("Hello from ai-learn!")
    information = """Elon Reeve Musk (sinh ngày 28 tháng 6 năm 1971) là một doanh nhân,
nổi tiếng với vai trò chủ chốt tại Tesla và SpaceX. Musk sinh ra và lớn lên ở
Pretoria, Nam Phi, trước khi di cư đến Canada. Ông chuyển tới California vào năm
1995. Musk cùng em trai Kimbal đồng sáng lập Zip2; sau đó ông đồng sáng lập X.com,
công ty về sau sáp nhập thành PayPal. Năm 2002, ông thành lập SpaceX."""
    summary_template = """Given the information below, please provide a concise summary in Vietnamese:

{information}

Summary:"""
    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
    if not api_key:
        raise RuntimeError("Hãy đặt ANTHROPIC_API_KEY hoặc CLAUDE_API_KEY trong file .env")

    workspace_id = os.getenv("ANTHROPIC_WORKSPACE_ID")
    default_headers = (
        {"anthropic-workspace-id": workspace_id} if workspace_id else None
    )
    llm = ChatAnthropic(
        model="claude-sonnet-5",
        api_key=api_key,
        default_headers=default_headers,
    )
    chain = summary_prompt_template | llm
    response = chain.invoke(
        {"information": information},
        config={
            "run_name": "vietnamese-summary",
            "tags": ["anthropic", "summary", "development"],
            "metadata": {"model": "claude-sonnet-5"},
        },
    )
    print("Summary:", response.content)


if __name__ == "__main__":
    main()
