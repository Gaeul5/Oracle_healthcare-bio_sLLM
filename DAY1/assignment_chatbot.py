import os
import json
from pathlib import Path

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

# --------------------------------------------------
# 환경변수 로드
# --------------------------------------------------
load_dotenv()

# --------------------------------------------------
# RAG 문서 로드 & 벡터스토어 생성
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
RAG_DOCS_DIR = Path(os.getenv("RAG_DOCS_DIR", BASE_DIR))
SUPPORTED_EXTENSIONS = {".txt", ".md", ".json"}
MAX_HISTORY_MESSAGES = 8
KEEP_RECENT_MESSAGES = 4

STYLE_MODES = {
    "normal": "친절하고 자연스러운 한국어로 답변하세요.",
    "easy": "초보자도 이해할 수 있게 쉬운 단어, 짧은 문장, 간단한 비유를 사용하세요.",
    "expert": "핵심 개념과 실무적인 주의점을 포함해 조금 더 전문적으로 답변하세요.",
}

OUTPUT_FORMATS = {
    "normal": "일반적인 대화형 답변으로 작성하세요.",
    "bullets": "핵심 내용을 불릿 목록으로 정리하세요.",
    "json": (
        '반드시 JSON 객체만 출력하세요. 형식은 '
        '{"answer": "답변", "key_points": ["핵심1", "핵심2"], "sources": ["출처"]} 입니다.'
    ),
}


def read_document(path: Path) -> str:
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return json.dumps(data, ensure_ascii=False, indent=2)

    return path.read_text(encoding="utf-8")


def split_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = max(end - overlap, start + 1)

    return chunks


def load_rag_documents(docs_dir: Path) -> list[Document]:
    if not docs_dir.exists():
        return []

    documents = []
    for path in sorted(docs_dir.rglob("*")):
        if not path.is_file() or path.suffix not in SUPPORTED_EXTENSIONS:
            continue

        try:
            text = read_document(path)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            print(f"[RAG] 문서를 건너뜁니다: {path.name} ({error})")
            continue

        relative_path = path.relative_to(docs_dir)
        for index, chunk in enumerate(split_text(text), start=1):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={"source": str(relative_path), "chunk": index},
                )
            )

    return documents


def build_retriever():
    documents = load_rag_documents(RAG_DOCS_DIR)
    if not documents:
        return None

    try:
        embeddings = OpenAIEmbeddings(
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        )
        vectorstore = InMemoryVectorStore.from_documents(documents, embeddings)
        return vectorstore.as_retriever(search_kwargs={"k": 3})
    except Exception as error:
        print(f"[RAG] 벡터스토어 생성에 실패했습니다: {error}")
        return None


def format_context(docs: list[Document]) -> str:
    if not docs:
        return "관련 문서를 찾지 못했습니다."

    formatted_docs = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        chunk = doc.metadata.get("chunk", "?")
        formatted_docs.append(f"[출처: {source} #{chunk}]\n{doc.page_content}")

    return "\n\n---\n\n".join(formatted_docs)


def format_history_for_summary(messages: list[HumanMessage | AIMessage]) -> str:
    lines = []
    for message in messages:
        speaker = "사용자" if isinstance(message, HumanMessage) else "챗봇"
        lines.append(f"{speaker}: {message.content}")

    return "\n".join(lines)


def print_help() -> None:
    print(
        "\n사용 가능한 명령어\n"
        "- /help: 명령어 보기\n"
        "- /clear: 대화 기록 초기화\n"
        "- /summary: 지금까지의 대화 요약\n"
        "- /easy: 쉬운 설명 모드\n"
        "- /expert: 전문가 모드\n"
        "- /normal: 기본 답변 모드\n"
        "- /bullets: 불릿 목록 출력\n"
        "- /json: JSON 출력\n"
        "- reload 또는 다시읽기: RAG 문서 다시 읽기\n"
        "- exit 또는 종료: 챗봇 종료\n"
    )


def summarize_dialogue(
    current_summary: str, messages: list[HumanMessage | AIMessage]
) -> str:
    if not messages:
        return current_summary

    return summary_chain.invoke(
        {
            "conversation_summary": current_summary,
            "new_dialogue": format_history_for_summary(messages),
        }
    )


# --------------------------------------------------
# 모델 & 프롬프트 (system + 대화 기록 + 이번 입력)
# --------------------------------------------------
model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0.5)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "당신은 친절한 한국어 학습 챗봇입니다. "
        "이전 대화와 검색된 참고 문서를 함께 참고해서 답변하세요. "
        "참고 문서에 근거한 내용은 답변 끝에 출처 파일명을 짧게 적으세요. "
        "참고 문서에 없는 내용은 모른다고 말하거나 일반 지식임을 구분하세요.\n\n"
        "[대화 요약]\n{conversation_summary}\n\n"
        "[답변 스타일]\n{style_instruction}\n\n"
        "[출력 형식]\n{format_instruction}\n\n"
        "[검색된 참고 문서]\n{context}",
    ),
    MessagesPlaceholder("history"),
    ("human", "{input}"),
])

chain = prompt | model | StrOutputParser()

summary_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "당신은 대화 내용을 짧고 정확하게 요약하는 도우미입니다. "
        "사용자의 목표, 이미 설명한 내용, 남은 질문을 한국어로 정리하세요.",
    ),
    (
        "human",
        "기존 요약:\n{conversation_summary}\n\n"
        "새 대화:\n{new_dialogue}\n\n"
        "업데이트된 요약:",
    ),
])

summary_chain = summary_prompt | model | StrOutputParser()
retriever = build_retriever()

# --------------------------------------------------
# 대화 기록 (직접 리스트로 관리)
# --------------------------------------------------
history: list[HumanMessage | AIMessage] = []
conversation_summary = "아직 요약된 대화가 없습니다."
style_mode = "normal"
output_format = "normal"

print("챗봇을 시작합니다. 종료하려면 'exit' 또는 '종료'를 입력하세요.")
print("명령어를 보려면 /help 를 입력하세요.")
if retriever:
    print(f"RAG 문서 폴더: {RAG_DOCS_DIR}")
    print("문서를 다시 읽으려면 'reload' 또는 '다시읽기'를 입력하세요.\n")
else:
    print(f"RAG 문서를 찾지 못했습니다. 문서 폴더: {RAG_DOCS_DIR}\n")

while True:
    user_input = input("나: ").strip()
    command = user_input.lower()

    if command in ("exit", "종료"):
        print("챗봇을 종료합니다.")
        break

    if command in ("reload", "다시읽기"):
        print("RAG 문서를 다시 읽는 중입니다...")
        retriever = build_retriever()
        if retriever:
            print("RAG 문서 재로드가 완료되었습니다.\n")
        else:
            print("RAG 문서를 찾지 못했습니다.\n")
        continue

    if command == "/help":
        print_help()
        continue

    if command == "/clear":
        history.clear()
        conversation_summary = "아직 요약된 대화가 없습니다."
        print("대화 기록과 요약을 초기화했습니다.\n")
        continue

    if command == "/summary":
        try:
            current_summary = summarize_dialogue(conversation_summary, history)
            print(f"\n현재 대화 요약:\n{current_summary}\n")
        except Exception as error:
            print(f"대화 요약을 만들지 못했습니다: {error}\n")
        continue

    if command == "/easy":
        style_mode = "easy"
        print("쉬운 설명 모드로 변경했습니다.\n")
        continue

    if command == "/expert":
        style_mode = "expert"
        print("전문가 모드로 변경했습니다.\n")
        continue

    if command == "/normal":
        style_mode = "normal"
        output_format = "normal"
        print("기본 답변 모드로 변경했습니다.\n")
        continue

    if command == "/bullets":
        output_format = "bullets"
        print("불릿 목록 출력으로 변경했습니다.\n")
        continue

    if command == "/json":
        output_format = "json"
        print("JSON 출력으로 변경했습니다.\n")
        continue

    if not user_input:
        continue

    try:
        retrieved_docs = retriever.invoke(user_input) if retriever else []
    except Exception as error:
        print(f"[RAG] 문서 검색에 실패했습니다: {error}")
        retrieved_docs = []
    context = format_context(retrieved_docs)

    print("봇: ", end="", flush=True)
    response_text = ""
    for chunk in chain.stream(
        {
            "input": user_input,
            "history": history,
            "conversation_summary": conversation_summary,
            "style_instruction": STYLE_MODES[style_mode],
            "format_instruction": OUTPUT_FORMATS[output_format],
            "context": context,
        }
    ):
        print(chunk, end="", flush=True)
        response_text += chunk
    print("\n")

    history.append(HumanMessage(content=user_input))
    history.append(AIMessage(content=response_text))

    if len(history) > MAX_HISTORY_MESSAGES:
        old_messages = history[:-KEEP_RECENT_MESSAGES]
        recent_messages = history[-KEEP_RECENT_MESSAGES:]
        try:
            conversation_summary = summarize_dialogue(conversation_summary, old_messages)
            history = recent_messages
        except Exception as error:
            print(f"[요약] 대화 요약에 실패했습니다: {error}")
