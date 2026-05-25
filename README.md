# K-FEDRI-v3a-
한국 FFDRI·캐나다 FWI·미국 NFDRS 개념을 참고한 임상·지형 보강형 ML 기반 산불위험 보조모델

K-FEDRI v3a는 “한국 FFDRI + 캐나다 FWI + 미국 NFDRS”의 공식 산식을 수학적으로 결합한 모델이 아니다. 더 정확히는 세 체계가 공통적으로 강조하는 위험 요인을 한국 공개자료 환경에서 구현 가능한 feature로 변환한 ML 기반 모델이다. 이 구분은 매우 중요하다. 공식 산식 구현이 아닌 proxy feature 기반 ML 모델이다.

# 라이브러리 필요 버전
streamlit>=1.32.0 
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.18.0
folium>=0.16.0
streamlit-folium>=0.18.0
