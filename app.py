#rich 라이브러리에서 console 모듈을 가져옵니다.
from rich.console import Console

#rich 라이브러리의 Console 객체를 생성합니다.
console = Console()

# rich 라이브러리의 Console 객체를 사용하여 콘솔에 메시지를 출력합니다.
console.print("Python 개발 환경 준비 완료!", style="bold green")
console.print("venv에서 라이브러리 설치 성공!", style="bold yellow")