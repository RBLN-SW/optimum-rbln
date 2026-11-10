import json
S = "<SCRATCH>/stage_d"
SETUP = (f"S={S}; mkdir -p $S/style_devtip && gh api repos/rebellions-sw/optimum-rbln/tarball/74e97fa4abff6868ab2f581a83fb78ddc0f23388 "
         "| tar -xz -C $S/style_devtip --strip-components=1")
hdr = json.load(open(f"{S}/scripts/style_header_devtip.json"))
notes = {"no_header": "저작권 헤더 없음", "rebellions_noncanonical_format": "헤더 형식이 표준과 다름(2024 단축형 또는 '#' 빈줄)", "year_mismatch": "헤더 연도가 파일 추가 연도와 다름"}
hv = [o for o in hdr if o["path"].startswith("src/") and o["kind"] in notes]
order = {"year_mismatch": 0, "rebellions_noncanonical_format": 1, "no_header": 2}
hv.sort(key=lambda o: (o["add_norename"] != "2026", order[o["kind"]], o["path"]))
hdr_viol = [{"path": o["path"], "line": 1, "snippet": o["first"][:120], "note": f"{notes[o['kind']]} (hdr {o['header_year']}, added {o['add_norename']})"} for o in hv[:25]]

res = [
 {"id": "STY-018",
  "check": {"kind": "script",
    "description": "dev 최신 트리에서 CI 게이트와 같은 `ruff format --check`/`ruff check`를 돌려 실패 파일을 세고, 2025-01 이후 머지 PR 중 머지 후 lint/docstring 실패를 고치는 후속 PR(제목 매칭)을 위반으로 센다.",
    "command": SETUP + "; cd $S/style_devtip && ruff format --no-cache --check . && ruff check --no-cache . ; python3 -c \"import json,re; ps=[json.loads(l) for l in open('$S/../corpus/prs.jsonl')]; ps=[p for p in ps if (p['mergedAt'] or '')>='2025-01']; print(len(ps)); [print(p['number'],p['title']) for p in ps if re.search(r'\\b(lint|ruff|format(ting)?|docstring)\\b',p['title'],re.I)]\""},
  "compliant": 627,
  "violations": [
    {"path": "PR #821", "line": None, "snippet": "other: fix the syntax error and lint failures left on dev by #733", "note": "#809·#813이 새 ruff 규칙 검사 없이 dev에 머지됨"},
    {"path": "PR #665", "line": None, "snippet": "other(model): fix lint in modeling modernbert", "note": "머지된 코드의 lint 실패를 후속 PR로 수정"},
    {"path": "PR #95", "line": None, "snippet": "Other(Seq2Seq): fix ruff formating for #68", "note": "#68 머지 후 ruff format 누락을 후속 수정"},
    {"path": "PR #467", "line": None, "snippet": "other: fix docstring fail cases", "note": "docstring 검사 실패를 후속 PR로 수정"}],
  "violations_total": 4,
  "verdict": "verified",
  "verdict_reason": "dev 최신(74e97fa4) 398개 파일과 worktree HEAD(main) 345개 파일 모두 ruff format/check를 통과하고, CI 게이트(.buildkite/pr.yml: ruff format·ruff check·mypy·check_inits·pre-commit, 별도 docstrings 단계; main은 check_code_quality.yml+test-docstrings.yml)가 머지 전에 강제한다. 2025-01 이후 631개 머지 PR 중 lint/docstring 후속 수정 PR은 4건뿐이다. mypy는 로컬에 설치돼 있지 않아 직접 실행하지 못했다.",
  "rule_refinement": "머지 전에 CI 품질 게이트(ruff format --check, ruff check, mypy, scripts/check_inits.py, mkdocstrings docstring 검사)를 통과시킨다. 다른 PR이 먼저 머지되어 base가 바뀌었으면(특히 lint 규칙을 바꾸는 PR 직후) 최신 base에서 CI를 다시 돌린 뒤 머지한다.",
  "suggested_test": "규칙 자체는 CI 게이트가 강제하므로 별도 pytest는 불필요하다. 회귀 방지를 원하면 tests/test_conventions.py에 .buildkite/pr.yml을 yaml로 읽어 'code quality' 단계 command에 `ruff format --check .`, `ruff check .`, `mypy`, `scripts/check_inits.py`가, 'docstrings' 단계가 존재하는지 확인하는 테스트를 둔다(허용 목록 없음)."},
 {"id": "STY-004",
  "check": {"kind": "ast",
    "description": "ruff F401/F841/F811/RUF059/PLW0127로 미사용 import·변수를 세고, AST로 docstring 없이 super()에 인자를 그대로 넘기기만 하는 override와 저장소 어디서도(src/tests/examples) 참조되지 않는 private(_name) 함수·메서드·클래스를 위반으로 센다.",
    "command": SETUP + "; cd $S/style_devtip && ruff check --no-cache --select F401,F841,F811,RUF059,PLW0127 --statistics . ; python3 $S/scripts/style_deadcode.py $S/style_devtip"},
  "compliant": 2154,
  "violations": [
    {"path": "src/optimum/rbln/diffusers/models/transformers/transformer_sd3.py", "line": 78, "snippet": "def __post_init__(self, **kwargs):", "note": "super().__post_init__(**kwargs)만 호출하는 부모와 동일한 override"},
    {"path": "src/optimum/rbln/transformers/modeling_attention_utils.py", "line": 425, "snippet": "def _required_memory_at(", "note": "정의 외 참조가 src/tests/examples 어디에도 없는 private classmethod"}],
  "violations_total": 2,
  "verdict": "verified",
  "verdict_reason": "dev 최신에서 ruff 미사용 import/변수 검사는 0건(__init__.py의 F401은 설정상 의도적 제외), 메서드 1685개 중 무의미한 pass-through override 1개(docstring을 붙이기 위한 12개는 제외), private 정의 471개 중 미참조 1개(_postprocess_outputs는 HF 훅이라 제외)뿐이다. 레거시 코드·효과 없는 컴파일 옵션·중복 CI 워크플로는 기계적으로 판정할 수 없어 측정하지 않았다.",
  "rule_refinement": "미사용 import·변수·unpack 변수는 ruff(F401, F841, RUF059)가, 부모와 동일한 override와 미참조 private 헬퍼는 리뷰가 잡는다. docstring만 덧붙이려는 pass-through override는 문서화 목적이므로 허용한다. '레거시 코드'·'효과 없는 컴파일 옵션'·'오래된 CI 워크플로'는 리뷰 판단 항목이다.",
  "suggested_test": "tests/test_conventions.py에 src/optimum/rbln 전체를 ast로 읽어, docstring·데코레이터(classmethod/staticmethod 제외) 없이 본문이 `[return] super().<같은이름>(<같은 인자 그대로>)` 한 줄뿐인 메서드를 찾는 테스트를 둔다. 허용 목록은 현재 위반 1건(transformer_sd3.py RBLNSD3Transformer2DModel.__post_init__)이며 수정 후 비운다. 미참조 private 검사는 HF 훅 오버라이드 오탐이 있어 테스트로 두지 않는다."},
 {"id": "STY-011",
  "check": {"kind": "ast",
    "description": "src/optimum/rbln의 모든 Subscript에서 typing Optional/Union/List/Dict/Tuple 사용과 인자에 문자열 forward-ref가 있는지, `X | None`·내장 제네릭 사용 수, 기본값 None인데 annotation이 None을 허용하지 않는 암묵적 Optional 파라미터, `from __future__ import annotations` 파일 수를 센다.",
    "command": SETUP + "; cd $S/style_devtip && python3 $S/scripts/style_typing.py src/optimum/rbln && python3 $S/scripts/style_typing_legacy.py src/optimum/rbln && python3 $S/scripts/style_typing_fwdref.py src/optimum/rbln"},
  "compliant": 1705,
  "violations": [
    {"path": "src/optimum/rbln/diffusers/pipelines/cosmos/pipeline_cosmos2_5_predict.py", "line": 52, "snippet": "safety_checker: RBLNCosmosSafetyChecker = None,", "note": "암묵적 Optional: 기본값 None인데 `| None` 없음"},
    {"path": "src/optimum/rbln/diffusers/pipelines/cosmos/pipeline_cosmos2_5_transfer.py", "line": 65, "snippet": "safety_checker: RBLNCosmosSafetyChecker = None,", "note": "암묵적 Optional: 기본값 None인데 `| None` 없음"},
    {"path": "src/optimum/rbln/diffusers/pipelines/cosmos/pipeline_cosmos2_text2image.py", "line": 49, "snippet": "safety_checker: RBLNCosmosSafetyChecker = None,", "note": "암묵적 Optional: 기본값 None인데 `| None` 없음"},
    {"path": "src/optimum/rbln/diffusers/pipelines/cosmos/pipeline_cosmos2_video2world.py", "line": 49, "snippet": "safety_checker: RBLNCosmosSafetyChecker = None,", "note": "암묵적 Optional: 기본값 None인데 `| None` 없음"}],
  "violations_total": 4,
  "verdict": "verified",
  "verdict_reason": "dev 최신에서 기본값 None인 annotated 파라미터 1709개 중 암묵적 Optional은 4개(main HEAD는 #736 이전이라 242개)이고, `X | None` 1832회·내장 제네릭 666회에 비해 남은 Optional/Union 86회는 전부 문자열 forward-ref를 감싼 경우다(문자열 없는 사용 0건). `from __future__ import annotations`는 1개 파일뿐이다. forward-ref에서는 `Optional[\"X\"]` 72회 대 `\"X | None\"` 17회로 typing 별칭이 우세하다.",
  "rule_refinement": "annotation은 `X | None`·`list[...]` 등 3.10+ 네이티브 문법으로 쓰고, 기본값이 None이면 반드시 `| None`을 명시한다. 단, 인자가 문자열 forward-ref여서 `|`로 쓸 수 없는 경우(`Optional[\"PreTrainedModel\"]`, `Union[dict[str, Any], \"RBLNModelConfig\"]`)는 예외로 남아 있다 — 이 경우 전체를 `\"X | None\"`처럼 문자열로 감싸는 쪽으로 통일할지는 팀 결정이 필요하다.",
  "suggested_test": "tests/test_conventions.py에 src/optimum/rbln 전체를 ast로 순회해 (1) 모든 함수 인자 중 기본값이 None 상수인데 annotation 문자열에 'None'/'Optional'/'Any'가 없는 것, (2) Optional/Union/List/Dict/Tuple Subscript 중 slice에 문자열 상수가 없는 것을 실패로 보고하는 테스트를 둔다. 허용 목록은 cosmos2 파이프라인 4개 파일의 safety_checker 파라미터. (mypy no_implicit_optional이 이미 핵심 모듈을 검사하지만 models.* 는 느슨하게 설정돼 있어 테스트가 보완한다.)"},
 {"id": "STY-013",
  "check": {"kind": "ast",
    "description": "src/optimum/rbln(utils/logging.py 제외)에서 stdlib `import logging`/`import warnings`, `logging.getLogger()`, `warnings.warn()` 호출과 transformers/diffusers logging import를 찾고, `utils.logging.get_logger`를 쓰는 파일 수와 비교한다.",
    "command": SETUP + "; cd $S/style_devtip && python3 $S/scripts/style_logging.py src/optimum/rbln ."},
  "compliant": 56,
  "violations": [
    {"path": "src/optimum/rbln/transformers/models/time_series_transformer/modeling_time_series_transformer.py", "line": 46, "snippet": "logger = logging.getLogger(__name__)", "note": "stdlib 로거 생성; get_logger 미사용"},
    {"path": "src/optimum/rbln/transformers/models/auto/auto_factory.py", "line": 181, "snippet": "warnings.warn(", "note": "UserWarning을 built-in warnings로 발행"},
    {"path": "src/optimum/rbln/utils/import_utils.py", "line": 149, "snippet": "warnings.warn(", "note": "ImportWarning을 built-in warnings로 발행"},
    {"path": "src/optimum/rbln/utils/import_utils.py", "line": 162, "snippet": "warnings.warn(", "note": "ImportWarning을 built-in warnings로 발행"},
    {"path": "src/optimum/rbln/utils/import_utils.py", "line": 179, "snippet": "warnings.warn(", "note": "ImportWarning을 built-in warnings로 발행"},
    {"path": "src/optimum/rbln/utils/import_utils.py", "line": 187, "snippet": "warnings.warn(", "note": "ImportWarning을 built-in warnings로 발행"}],
  "violations_total": 6,
  "verdict": "verified",
  "verdict_reason": "로그를 남기는 56개 파일이 `utils.logging.get_logger`를 쓰고, stdlib 로거를 직접 만드는 곳은 time_series_transformer 1곳뿐이다. built-in warnings.warn은 3개 파일 6곳(그중 4곳은 패키지 import 시 버전 호환성 ImportWarning)으로 소수다. modeling_attention_utils.py의 `import logging`은 `logging.DEBUG` 상수 참조용이라 위반에서 뺐다.",
  "rule_refinement": "로거는 항상 `from ...utils.logging import get_logger`로 만들고 stdlib `logging.getLogger`를 쓰지 않는다(`logging.DEBUG` 같은 레벨 상수 참조는 허용). 사용자 경고도 `logger.warning`/`warning_once`로 낸다. 단 import 시점의 버전 호환성 경고(`warnings.warn(..., ImportWarning)`)를 예외로 둘지는 팀 확인이 필요하다.",
  "suggested_test": "tests/test_conventions.py에 src/optimum/rbln/**/*.py(utils/logging.py 제외)를 ast로 읽어 `logging.getLogger(...)` 호출과 `warnings.warn(...)` 호출을 실패로 보고하는 테스트를 둔다. 허용 목록: modeling_time_series_transformer.py:46(수정 대상), auto_factory.py의 warnings.warn 1곳, utils/import_utils.py의 ImportWarning 4곳(팀이 예외로 확정하면 영구 허용)."},
 {"id": "STY-005",
  "check": {"kind": "script",
    "description": "src/·tests/의 모든 .py 첫 줄(shebang 제외)과 이어지는 13줄을 표준 Rebellions Apache 헤더와 비교하고, 헤더 연도를 `git log --diff-filter=A`(--follow/--no-renames 둘 다)로 구한 추가 연도와 대조한다; 헤더 없음·형식 불일치·연도 불일치를 위반으로 센다(HF 유래 헤더, 빈 파일, 2023 초기 import 후 #19에서 2025로 일괄 갱신된 파일은 준수로 본다).",
    "command": SETUP + "; python3 $S/scripts/style_header_tree.py $S/style_devtip origin/dev 2026 $S/scripts/style_header_devtip.json  # origin/dev(ae60d5fd)에 없는 파일은 2026 추가로 간주"},
  "compliant": 290,
  "violations": hdr_viol,
  "violations_total": 47,
  "verdict": "aspirational",
  "verdict_reason": "dev 최신 src의 Rebellions 소유 파일 337개 중 290개(86%)만 표준 헤더와 맞는 연도를 갖고, 헤더 없음 11·형식 불일치 20·연도 불일치 16건이 있다. 2026년에 새로 추가된 src 파일만 봐도 78개 중 59개(76%)만 준수해 새 파일에서도 형식('#' 빈줄 변형 9개)과 연도(2025 복사 5개) 이탈이 계속된다. tests/는 12개 중 헤더가 있는 파일이 0개다.",
  "rule_refinement": "src/optimum/rbln 아래 새 파일은 첫 줄 `# Copyright <추가 연도> Rebellions Inc. All rights reserved.`, 빈 줄, 그리고 기존 파일과 바이트 단위로 같은 Apache-2.0 블록(주석 사이 빈 줄은 '#'이 아닌 진짜 빈 줄)을 붙인다. HuggingFace 코드를 옮긴 파일은 원 저작권 헤더를 유지하고, tests/는 현재 헤더를 붙이지 않는 것이 관행이다(적용 범위 확인 필요).",
  "suggested_test": "tests/test_conventions.py에 src/optimum/rbln/**/*.py 중 비어 있지 않은 파일의 첫 13줄(shebang 제외)이 정규식 `# Copyright 20\\d\\d Rebellions Inc. All rights reserved.` + 표준 Apache 블록과 정확히 일치하는지 검사하는 테스트를 둔다(서드파티 헤더 파일 9개는 'Copyright .* (HuggingFace|Optuna)' 로 면제). 연도-추가일 대조는 git 이력이 필요해 테스트에서 빼고, 현재 형식 위반·헤더 없음 31개 파일을 허용 목록으로 시작해 줄여 간다."},
 {"id": "STY-010",
  "check": {"kind": "ast",
    "description": "__init__.py를 제외한 모듈의 `if TYPE_CHECKING:` 본문 import 이름마다 모듈 수준 런타임 import와의 중복, annotation 밖 런타임 사용, `from __future__` 없이 따옴표 없는 annotation 사용, 미사용 여부를 검사하고, pyproject의 ruff ignore에 TC001~TC003이 있는지 확인한다; first-party TYPE_CHECKING import가 실제 순환을 피하는지도 정적 import 그래프로 시도했다.",
    "command": SETUP + "; cd $S/style_devtip && python3 $S/scripts/style_type_checking.py src/optimum/rbln . && python3 $S/scripts/style_tc_cycle.py src && grep -n 'TC00' pyproject.toml"},
  "compliant": 103,
  "violations": [
    {"path": "src/optimum/rbln/utils/submodule.py", "line": 119, "snippet": "submodule_cls: type[RBLNModel] = get_rbln_model_cls(submodule_rbln_config.rbln_model_cls_name)", "note": "TYPE_CHECKING 이름을 따옴표 없이 사용(지역 변수라 런타임 무해)"}],
  "violations_total": 1,
  "verdict": "verified",
  "verdict_reason": "dev 최신에서 53개 모듈의 TYPE_CHECKING 이름 104개 중 런타임 중복 import는 0건(#733 직전 origin/dev에는 15개 모듈 18건)이고, annotation 밖 런타임 사용은 지역 재-import 1건뿐이며, 따옴표 없는 사용은 평가되지 않는 지역 변수 annotation 1건이다. pyproject는 TC001~TC003·TC006을 ignore하고 TC004·TC005만 켠다. '순환 참조이거나 외부 타입일 때만'이라는 조건은 정적 그래프가 #556의 실제 순환을 재현하지 못해 코드로 판정하지 못했다(TC import 104개 중 third-party 73, first-party 31).",
  "rule_refinement": "annotation 전용 import를 `if TYPE_CHECKING:`으로 옮길 때는 annotation을 문자열로 감싸고, 같은 이름을 모듈 수준 런타임 import와 중복하지 않는다; ruff TC001~TC003/TC006은 끄고 TC004/TC005만 켠다. first-party 심볼을 TYPE_CHECKING으로 옮기는 근거(순환 회피)는 PR 설명에 적는 판단 항목이다.",
  "suggested_test": "tests/test_conventions.py에 src/optimum/rbln의 __init__.py가 아닌 모듈을 ast로 읽어 `if TYPE_CHECKING:` 본문에서 import한 이름이 같은 모듈의 최상위 런타임 import에도 있으면 실패하는 테스트를 둔다(허용 목록 없음; TC004가 반대 방향을 이미 막는다). 추가로 pyproject.toml의 [tool.ruff.lint] ignore에 TC001·TC002·TC003이 있고 select에 TC가 있는지 확인한다."},
 {"id": "STY-012",
  "check": {"kind": "ast",
    "description": "모든 모듈의 `__all__`, 패키지 __init__.py의 TYPE_CHECKING 블록 `from .x import *` 순서, (main의) `_import_structure` 키·값 목록, 그리고 문자열 3개 이상을 담은 모듈 수준 list/dict를 sorted()/RUF022(isort 스타일)/대소문자 무시 정렬 중 하나와 일치하는지 검사한다.",
    "command": SETUP + "; cd $S/style_devtip && ruff check --no-cache --select RUF022,RUF023 . && python3 $S/scripts/style_sorted.py src/optimum/rbln ."},
  "compliant": 264,
  "violations": [],
  "violations_total": 0,
  "verdict": "verified",
  "verdict_reason": "dev 최신에서 `__all__` 185개와 __init__ TYPE_CHECKING star-import 블록 79개가 전부 정렬돼 있고 RUF022와 scripts/check_inits.py가 이를 강제한다. 정렬되지 않은 모듈 상수 7개(log_levels, _BYTE_UNITS, RBLN_VERSION_COMPATS, ROPE_INIT_FUNCTIONS, QUANTIZED_WEIGHTS 등)는 의미 순서를 따르는 표라 대상이 아니다. main HEAD에는 #733 이전 `_import_structure` 형식이라 미정렬 목록 37개가 남아 있다.",
  "rule_refinement": "export·등록 목록(`__all__`, 패키지 __init__의 re-export, 모델/설정 등록 목록)은 정렬 순서를 유지한다(RUF022 isort 스타일). 로그 레벨·단위·버전 호환표처럼 순서 자체가 의미인 상수 표는 예외다.",
  "suggested_test": "RUF022와 scripts/check_inits.py가 CI에서 이미 강제하므로 별도 테스트는 선택 사항이다. 둔다면 tests/test_conventions.py에서 src/optimum/rbln/**/__init__.py의 `if TYPE_CHECKING:` 본문 star-import 모듈 이름 목록이 sorted()와 같은지 확인한다(허용 목록 없음)."},
]
out = f"{S}/out/batch_style.json"
json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)
json.load(open(out))
for r in res: print(r["id"], r["verdict"], f"{r['compliant']}/{r['violations_total']}", len(r["violations"]))
