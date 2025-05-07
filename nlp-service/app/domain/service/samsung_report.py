from konlpy.tag import Okt
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import os
import logging

logger = logging.getLogger("nlp_api")

class SamsungReport:
    def __init__(self):
        self.text = ""
        self.tokens = []
        self.nouns = []
        self.stopwords = []
        self.word_count = {}
        print("🚀 SamsungReport 초기화 완료")

    def read_file(self):
        print("📖 파일 읽기 시작...")
        file_path = os.path.join('app', 'orginal', 'kr-Report_2018.txt')
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.text = f.read()
            print(f"✅ 파일 읽기 완료: {len(self.text)} 글자")
        except Exception as e:
            print(f"❌ 파일 읽기 오류: {e}")
            self.text = ""
        return self.text

    def extract_hangul(self):
        print("🔍 한글 추출 시작...")
        # 한글과 공백만 남기고 모두 제거
        original_length = len(self.text)
        self.text = re.sub('[^ㄱ-ㅎㅏ-ㅣ가-힣 ]', '', self.text)
        print(f"✅ 한글 추출 완료: {original_length} -> {len(self.text)} 글자")
        return self.text

    def change_token(self):
        print("🔄 토큰화 시작...")
        # 텍스트를 토큰으로 분리
        self.tokens = self.text.split()
        print(f"✅ 토큰화 완료: {len(self.tokens)} 개의 토큰")
        return self.tokens

    def extract_noun(self):
        print("📝 형태소 분석 시작...")
        okt = Okt()
        self.nouns = []
        total_tokens = len(self.tokens)
        
        for i, token in enumerate(self.tokens, 1):
            if i % 100 == 0:
                print(f"⏳ 형태소 분석 진행중: {i}/{total_tokens} ({i/total_tokens*100:.1f}%)")
            # token 내 단어별 품사 확인
            pos_tags = okt.pos(token)
            # 명사만 추출
            self.nouns.extend([word for word, tag in pos_tags if tag == 'Noun'])
        
        print(f"✅ 형태소 분석 완료: {len(self.nouns)} 개의 명사 추출")
        return self.nouns

    def read_stopword(self):
        print("📚 불용어 사전 읽기 시작...")
        # stopwords.txt 파일에서 불용어 읽기
        file_path = os.path.join('app', 'orginal', 'stopwords.txt')
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # 파일의 내용을 읽어와서 공백으로 분리
                words = f.read().strip().split()
                # 중복 제거 및 길이가 1인 단어 제외
                self.stopwords = list(set([word for word in words if len(word) > 1]))
            print(f"✅ 불용어 사전 읽기 완료: {len(self.stopwords)} 개의 불용어")
        except Exception as e:
            print(f"❌ 불용어 사전 읽기 오류: {e}")
            # 파일 읽기 실패시 기본 불용어 설정
            self.stopwords = ['및', '등', '㈜', '것', '저', '월', '년', '분기', '억원', '조원']
            print("⚠️ 기본 불용어 설정 적용")
        
        return self.stopwords

    def remove_stopword(self):
        print("🗑️ 불용어 제거 시작...")
        if not self.stopwords:
            self.read_stopword()

        # 1. 기본 필터링
        filtered = [noun for noun in self.nouns if len(noun) > 1 and not noun.isdigit()]
        print(f"1️⃣ 기본 필터링 완료: {len(self.nouns)} -> {len(filtered)}")

        # 2. 빈도 기반 자동 필터링
        counter = Counter(filtered)
        total = sum(counter.values())
        max_freq_ratio = 0.03
        min_freq = 3
        auto_stopwords = {
            word for word, freq in counter.items()
            if freq / total >= max_freq_ratio or freq < min_freq
        }
        print(f"2️⃣ 빈도 기반 필터링: {len(auto_stopwords)} 개의 단어 제거 예정")

        # 3. 의미기반 불용어 추가
        SEMANTIC_STOPWORDS = {
            "각주", "사의", "제시", "가능", "사항", "내용", "경우", "출연", "연장",
            "부분", "대상", "상황", "수준", "관련", "조치", "결과", "이상", "체계",
            "방안", "사안", "대비", "기준", "대응", "확보", "실현", "수립", "고려",
            "계획", "항목", "기타", "포함", "조정", "연계", "적용", "검토"
        }

        all_stopwords = set(self.stopwords) | auto_stopwords | SEMANTIC_STOPWORDS
        self.nouns = [noun for noun in filtered if noun not in all_stopwords]

        print(f"✅ 불용어 제거 완료: {len(filtered)} -> {len(self.nouns)} 개의 단어")
        print(f"[자동 제거된 불용어 예시] {list(auto_stopwords)[:10]}")
        return self.nouns

    def find_frequency(self):
        print("📊 단어 빈도 분석 시작...")
        # 단어 빈도수 계산
        self.word_count = Counter(self.nouns)
        result = dict(self.word_count.most_common(100))
        print(f"✅ 단어 빈도 분석 완료: 상위 10개 단어 {list(result.items())[:10]}")
        return result

    def draw_wordcloud(self):
        print("🎨 워드클라우드 생성 시작...")
        # matplotlib 백엔드를 Agg로 설정 (GUI 없이 동작)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        try:
            # 워드클라우드 생성
            wc = WordCloud(
                font_path='/usr/share/fonts/truetype/nanum/NanumGothic.ttf',  # 나눔고딕 폰트 경로
                background_color='white',
                width=800,
                height=600,
                max_words=100,
                max_font_size=200
            )
            print("1️⃣ 워드클라우드 객체 생성 완료")
            
            # 워드클라우드 생성
            wc.generate_from_frequencies(self.word_count)
            print("2️⃣ 워드클라우드 데이터 생성 완료")
            
            # 이미지 저장 - app/output 폴더에 저장
            output_dir = os.path.join('app', 'output')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"3️⃣ 출력 디렉토리 생성: {output_dir}")
            
            plt.figure(figsize=(10, 8))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis('off')
            
            output_path = os.path.join(output_dir, 'samsung_wordcloud.png')
            plt.savefig(output_path)
            plt.close('all')  # 모든 figure 닫기
            
            print(f"✅ 워드클라우드 생성 완료: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ 워드클라우드 생성 오류: {e}")
            raise e
