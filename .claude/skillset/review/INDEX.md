# optimum-rbln 룰 리뷰 시트

생성일 2026-10-26. 출처: PR 804건·리뷰 코멘트 1,037건·대화 코멘트 375건에서 채굴한 뒤 코드(dev tip)와 대조한 룰. 각 룰은 **룰 · 근거(원문 인용, PR 링크) · 코드 대조 · 응답** 네 줄이다. 판단형은 코드로 검사할 수 없어 대조를 건너뛴 룰이다. 리뷰 방법은 [README.md](README.md).

| 영역 | 파일 | 결정 필요 | 코드 일치 | 코드 불일치 | 판단형 | 폐기 제안 | 합계 |
|---|---|---|---|---|---|---|---|
| rbln_config (설정 객체) | [rbln_config.md](rbln_config.md) | 4 | 1 | 1 | 11 | 1 | 18 |
| 멀티모달 · 서브모듈 | [multimodal_submodule.md](multimodal_submodule.md) | 1 | 1 | 2 | 4 | 1 | 9 |
| 모델 베이스 · 라이프사이클 | [modeling_base_lifecycle.md](modeling_base_lifecycle.md) | 4 | 3 | 0 | 9 | 1 | 17 |
| decoderonly | [decoderonly.md](decoderonly.md) | 0 | 1 | 0 | 2 | 0 | 3 |
| attention · KV cache | [attention_kvcache.md](attention_kvcache.md) | 1 | 0 | 0 | 2 | 2 | 5 |
| diffusers | [diffusers.md](diffusers.md) | 1 | 0 | 0 | 5 | 0 | 6 |
| 공개 API · 이름 | [api_naming.md](api_naming.md) | 0 | 4 | 0 | 5 | 1 | 10 |
| 테스트 | [testing.md](testing.md) | 0 | 2 | 1 | 20 | 0 | 23 |
| docstring · 문서 | [docstring_docs.md](docstring_docs.md) | 1 | 2 | 0 | 3 | 0 | 6 |
| 의존성 · 버전 | [dependency_version.md](dependency_version.md) | 0 | 6 | 1 | 4 | 2 | 13 |
| PR · CI · 브랜치 운영 | [pr_process.md](pr_process.md) | 0 | 4 | 3 | 14 | 4 | 25 |
| 코드 스타일 · 기타 | [style_other.md](style_other.md) | 1 | 6 | 1 | 5 | 0 | 13 |
| **합계** | | 13 | 30 | 9 | 84 | 12 | 148 |
