# 🎮 PuertoRico-BoardGame-RL-Balancing 협업 가이드

이 문서는 `dae-hany/PuertoRico-BoardGame-RL-Balancing` 프로젝트의 효율적인 협업을 위한 규칙을 정의합니다. 모든 팀원은 작업 시작 전 이 내용을 숙지해 주세요.

---

## 1. 브랜치 전략 (GitHub Flow)

`main` 브랜치는 언제나 실행 가능한 상태를 유지해야 합니다. 모든 새로운 작업은 별도의 브랜치에서 진행합니다.

- **`main`**: 제품 출시 및 배포 가능한 수준의 안정된 코드.
- **`feature/기능명`**: 새로운 기능 개발 (예: `feature/board-logic`, `feature/rl-agent`)
- **`fix/버그명`**: 버그 수정 (예: `fix/action-space-error`)
- **`docs/문서명`**: 문서 수정 (예: `docs/update-readme`)

---

## 2. 커밋 메시지 규칙 (Commit Convention)

누가 어떤 작업을 했는지 명확히 알기 위해 아래 형식을 따릅니다.

> **형식:** `[타입] 요약 내용 (#이슈번호)`

- **`Feat`**: 새로운 기능 추가
- **`Fix`**: 버그 수정
- **`Docs`**: 문서 수정 (README, CONTRIBUTING 등)
- **`Style`**: 코드 포맷팅, 세미콜론 누락 등 (코드 변경 없음)
- **`Refactor`**: 코드 리팩토링
- **`Test`**: 테스트 코드 추가 및 수정
- **`Chore`**: 빌드 업무 수정, 패키지 매니저 수정 등

**예시:** `[Feat] 강화학습 에이전트 초기 모델 구현 (#1)`

---

## 3. Pull Request (PR) 및 코드 리뷰 절차

모든 코드는 PR을 통해 `main` 브랜치에 합쳐집니다.

1. **Issue 생성**: 작업할 내용을 Issue에 등록하고 담당자를 지정합니다.
2. **Branch 생성**: 로컬에서 `feature/` 브랜치를 생성합니다.
3. **작업 및 Push**: 작업을 완료하고 원격 레포지토리에 push합니다.
4. **PR 생성**: GitHub에서 PR을 생성합니다. (내용에 작업 내용 요약 및 관련 Issue 번호 기재)
5. **Code Review**: 최소 1명 이상의 팀원에게 승인(Approve)을 받아야 합니다.
6. **Merge**: 리뷰가 완료되면 `Squash and merge` 방식으로 합칩니다.

---

## 4. 파이썬 환경 관리

팀원 간 라이브러리 버전을 맞추기 위해 아래 규칙을 지킵니다.

- 새로운 라이브러리를 설치한 경우, 반드시 `requirements.txt`를 업데이트합니다.
  ```bash
  pip freeze > requirements.txt
  ```
- 코드를 내려받은 후에는 항상 환경을 업데이트합니다.
  ```bash
  pip install -r requirements.txt
  ```
