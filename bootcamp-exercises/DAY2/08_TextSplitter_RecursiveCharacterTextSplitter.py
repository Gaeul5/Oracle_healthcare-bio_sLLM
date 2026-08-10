from langchain_text_splitters import RecursiveCharacterTextSplitter
sample_text = """한강 작가가 차린 독립서점 ‘책방오늘’이 오는 7일 폐업한다. 2018년 처음 문을 연 지 8년 만이다.

책방오늘은 최근 인스타그램을 통해 “‘책방오늘’이 양재동을 떠나 서촌 통의동의 골목에서 손님들을 맞이한 지 꼭 3년이 되는 2026년 7월7일, 이 공간에서의 마지막 영업을 하게 되었다”며 “열평 남짓한 공간을 임대하고 수선해 불을 밝히고, 책들을 들여 손님들과 만나고, 계절마다 ‘작가의 서가‘를 소개하고, 낭독회와 워크숍들을 열며 좋은 분들과 함께할 수 있어 의미 깊었던 시간이었다”고 안내했다. 영업을 재개할 여지를 남겨두되, “다시 문을 열게 될 시기와 장소는 아직 정해지지 않았다”고 책방오늘 쪽은 밝혔다.

책방오늘은 2018년 7월부터 준비해 두달 뒤인 9월 서울 서초구 양재에서 개시한 뒤 2023년 종로로 이전했다. 한강 작가의 독립서점에 대한 애정은 익히 알려져 있다. 실제 큐레이션을 직접 맡고, 북토크 등을 기획하고 작가를 초대하기도 했다. 한강은 2016년 영국 매체와의 인터뷰에서 글쓰기를 할 수 없다면 “서울 외곽에 작은 독립서점을 운영하고 싶다”고 밝힌 적도 있다. 다만 2024년 10월 노벨 문학상 수상자로 결정되면서 10평 규모의 책방 안팎으로 손님들이 대거 몰려 11월부터는 “(한강 작가는) 책방오늘의 운영에 더 이상 관여하지 않”는다고 알려야 했다. 이후 이사직으로 물러나 있던 것으로 전해진다. 
"""
for size in [50, 150, 300]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=int(size * 0.2),
        separators=["\n\n", "\n", ". ", " ", ""],
        )
    chunks = text_splitter.split_text(sample_text)

    print("=" * 70)
    print(f"chunk_size={size}, chunk_overlap={int(size * 0.2)}, chunk_count={len(chunks)}")

for i, chunk in enumerate(chunks):
    print(f"{i+1}번 청크, 길이={len(chunk)}")
    print(repr(chunk))
    print()