"""
instance_manager.py
===================
사용자가 UI를 통해 추가하는 검토건(Investment) 인스턴스를 RDF 그래프에 동적으로 추가/삭제하는 모듈.

설계 원칙:
1. 한글 입력 → 영어 IRI 자동 변환 (Demo_User_XXX 네임스페이스)
2. 세션별 격리: 사용자가 추가한 인스턴스 ID 목록 별도 관리
3. 안전장치: 펀드명 한 번에 1개만, 분기 최대 3개
4. v1.10 온톨로지 스키마와 일관 (BusinessProcess, BranchState, ProductType 등)
"""

import re
import uuid
from datetime import datetime
from rdflib import Graph, Namespace, RDF, RDFS, Literal, URIRef
from rdflib.namespace import XSD

INV = Namespace("http://company.com/investment-ontology#")

# ============================================
# 단계·상태·상품 옵션 (v1.10 기준)
# ============================================

# v1.10 실제 정의된 Stage (4개)
# ※ 추가 단계 필요 시 온톨로지 보강 후 추가
STAGES_FULL = [
    ("PreliminaryScreening", "사전협의신청관리"),
    ("PreliminaryReview", "예비검토내역관리"),
    ("LimitCommitment", "한도약정등록"),
    ("DrawdownProposal", "개별품의등록"),
]

# 축약(옵션 2 디폴트)도 동일 - 4개뿐이라 옵션 1과 같음
STAGES_SIMPLE = STAGES_FULL

# v1.10 실제 정의된 State (9개)
STATES_FULL = [
    ("State_ScreeningInProgress", "사전협의진행중", 0),
    ("State_ReviewInProgress", "예비검토진행중", 0),
    ("State_ReviewStalled", "예비검토단계정체", -1),
    ("State_CommitmentInProgress", "약정진행중", 1),
    ("State_CommitmentCompleted", "약정체결완료", 2),
    ("State_DrawdownApprovalInProgress", "개별출자품의진행중", 3),
    ("State_DrawdownApproved", "개별출자승인완료", 4),
    ("State_FundDisbursementCompleted", "출자금납입완료", 5),
    ("State_PostManagement", "사후관리진행", 6),
]

# 축약(옵션 2 디폴트) - 자주 쓰는 5개만
STATES_SIMPLE = [
    ("State_ReviewInProgress", "검토 진행 중", 0),
    ("State_ReviewStalled", "검토 정체", -1),
    ("State_CommitmentInProgress", "약정 진행 중", 1),
    ("State_CommitmentCompleted", "약정 완료", 2),
    ("State_FundDisbursementCompleted", "출자금 납입 완료", 5),
]

# 상품 유형 (v1.10 실제 정의된 Product 인스턴스 기반)
# LP출자는 펀드 형태(PEF/VentureFund/NewTechFund 등)로 표현하고, 인수금융은 별도
PRODUCT_TYPES = [
    ("Product_PEF", "LP출자 - 사모투자합자회사 (PEF)"),
    ("Product_VentureFund", "LP출자 - 벤처투자조합"),
    ("Product_NewTechFund", "LP출자 - 신기술사업투자조합"),
    ("Product_LimitedPartnership", "LP출자 - 투자합자조합"),
    ("Product_InvestmentTrust", "LP출자 - 투자신탁"),
    ("Product_AcquisitionFinance", "인수금융"),
]

# 펀드 형태 (RecipientFund의 type으로 사용)
# v1.10에서는 RecipientFund 하나만 있어서 단일 type 사용
# 펀드 형태 차이는 hasProductType (Branch에서) 으로 표현
FUND_TYPES = [
    ("RecipientFund", "출자대상펀드"),  # 단일 type
]


# ============================================
# 한글 → 영어 IRI 변환
# ============================================

def korean_to_iri_safe(text):
    """
    한글 텍스트를 IRI에 안전한 영어로 변환.
    예: "삼성펀드" -> "samsung_fund_a3f2b1"
    
    완벽한 한영 변환은 어려우므로 ASCII 안전 문자만 추출하고
    UUID 6자리를 붙여 고유성 보장.
    """
    # ASCII 영문/숫자만 추출
    ascii_part = re.sub(r'[^a-zA-Z0-9]', '', text)
    
    # 영문이 아예 없으면 'fund'
    if not ascii_part:
        ascii_part = 'fund'
    
    # 길이 제한
    ascii_part = ascii_part[:20].lower()
    
    # 짧은 UUID 추가 (충돌 방지)
    suffix = uuid.uuid4().hex[:6]
    
    return f"{ascii_part}_{suffix}"


# ============================================
# 인스턴스 추가 / 삭제
# ============================================

def add_user_investment(g, fund_name_kr, gp_name_kr, fund_type_iri, branches_data):
    """
    사용자가 추가한 검토건을 그래프에 추가.
    
    Args:
        g: rdflib Graph (메모리 그래프)
        fund_name_kr: 사용자가 입력한 펀드명 (한글)
        gp_name_kr: 사용자가 입력한 GP명 (한글)
        fund_type_iri: 펀드 형태 IRI (예: "PEF")
        branches_data: 분기 정보 리스트 [{"product_iri":..., "amount":..., "stage_iri":..., "state_iri":...}, ...]
    
    Returns:
        dict: {
            "investment_iri": "Demo_UserInvestment_xxx",
            "fund_iri": "Demo_UserFund_xxx",
            "gp_iri": "Demo_UserGP_xxx",
            "branch_iris": [...],
            "added_triples": [...]  # 추가된 트리플 (롤백용)
        }
    """
    # IRI 생성
    fund_id = korean_to_iri_safe(fund_name_kr)
    gp_id = korean_to_iri_safe(gp_name_kr)
    
    investment_iri = URIRef(f"{INV}Demo_UserInvestment_{fund_id}")
    fund_iri = URIRef(f"{INV}Demo_UserFund_{fund_id}")
    gp_iri = URIRef(f"{INV}Demo_UserGP_{gp_id}")
    fund_type = URIRef(f"{INV}{fund_type_iri}")
    
    added_triples = []
    
    def add(s, p, o):
        g.add((s, p, o))
        added_triples.append((s, p, o))
    
    # 1. GP 인스턴스
    add(gp_iri, RDF.type, INV.NamedIndividual)
    add(gp_iri, RDF.type, INV.GP)
    add(gp_iri, RDF.type, INV.Counterparty)
    add(gp_iri, RDF.type, INV.Customer)
    add(gp_iri, RDFS.label, Literal(gp_name_kr, lang="ko"))
    add(gp_iri, RDFS.comment, Literal(f"사용자 추가 GP (세션 시연용): {gp_name_kr}", lang="ko"))
    
    # 2. Fund 인스턴스
    add(fund_iri, RDF.type, INV.NamedIndividual)
    add(fund_iri, RDF.type, fund_type)
    add(fund_iri, RDF.type, INV.RecipientFund)
    add(fund_iri, RDF.type, INV.Counterparty)
    add(fund_iri, RDF.type, INV.Customer)
    add(fund_iri, RDFS.label, Literal(fund_name_kr, lang="ko"))
    add(fund_iri, RDFS.comment, Literal(f"사용자 추가 펀드 (세션 시연용): {fund_name_kr}", lang="ko"))
    add(fund_iri, INV.hasManager, gp_iri)
    
    # 3. Investment 인스턴스
    add(investment_iri, RDF.type, INV.NamedIndividual)
    add(investment_iri, RDF.type, INV.Investment)
    add(investment_iri, RDFS.label, Literal(f"{fund_name_kr} 검토 건", lang="ko"))
    add(investment_iri, RDFS.comment, Literal(
        f"사용자가 직접 등록한 검토 건 ({datetime.now().strftime('%H:%M')} 등록)", lang="ko"
    ))
    add(investment_iri, INV.managedByGP, gp_iri)
    add(investment_iri, INV.hasIndirectTarget, fund_iri)
    
    # 4. 분기 인스턴스들
    branch_iris = []
    for idx, branch in enumerate(branches_data, start=1):
        branch_iri = URIRef(f"{INV}Demo_UserBranch_{fund_id}_B{idx:02d}")
        product_iri = URIRef(f"{INV}{branch['product_iri']}")
        stage_iri = URIRef(f"{INV}{branch['stage_iri']}")
        state_iri = URIRef(f"{INV}{branch['state_iri']}")
        amount = int(branch['amount']) * 100000000  # 억원 → 원 변환
        
        product_label = next((label for iri, label in PRODUCT_TYPES if iri == branch['product_iri']), branch['product_iri'])
        
        add(branch_iri, RDF.type, INV.NamedIndividual)
        add(branch_iri, RDF.type, INV.InvestmentBranch)
        add(branch_iri, RDFS.label, Literal(f"{fund_name_kr} 분기 {idx} ({product_label})", lang="ko"))
        add(branch_iri, INV.hasProductType, product_iri)
        add(branch_iri, INV.hasInvestmentAmount, Literal(amount, datatype=XSD.integer))
        add(branch_iri, INV.hasCurrentStage, stage_iri)
        add(branch_iri, INV.hasBranchState, state_iri)
        
        # Investment → Branch 연결
        add(investment_iri, INV.hasBranch, branch_iri)
        
        branch_iris.append(str(branch_iri))
    
    return {
        "investment_iri": str(investment_iri).split('#')[-1],
        "fund_iri": str(fund_iri).split('#')[-1],
        "gp_iri": str(gp_iri).split('#')[-1],
        "fund_name_kr": fund_name_kr,
        "gp_name_kr": gp_name_kr,
        "branch_iris": [b.split('#')[-1] for b in branch_iris],
        "added_triples": added_triples,
        "n_branches": len(branches_data),
    }


def remove_user_investment(g, instance_record):
    """
    사용자가 추가했던 인스턴스를 그래프에서 제거.
    add_user_investment가 반환한 instance_record를 받아 added_triples를 모두 제거.
    """
    for triple in instance_record["added_triples"]:
        g.remove(triple)


def remove_all_user_investments(g, instance_records):
    """모든 사용자 추가 인스턴스 제거"""
    for record in instance_records:
        remove_user_investment(g, record)


# ============================================
# 사용자 추가 인스턴스 검색
# ============================================

def find_user_investment_by_keyword(instance_records, keyword):
    """
    사용자가 추가한 인스턴스 중 펀드명이나 GP명에 키워드가 포함된 것을 찾음.
    예: 사용자가 "삼성펀드"를 추가했고 질문이 "삼성펀드 어디까지?"면 매칭.
    """
    keyword_lower = keyword.lower().strip()
    
    for record in instance_records:
        fund_name = record["fund_name_kr"].lower()
        gp_name = record["gp_name_kr"].lower()
        
        # 양방향 부분 매칭
        if keyword_lower in fund_name or fund_name in keyword_lower:
            return record
        if keyword_lower in gp_name or gp_name in keyword_lower:
            return record
    
    return None


def list_all_user_investments(instance_records):
    """등록된 사용자 인스턴스 목록 (UI 표시용)"""
    return [
        {
            "investment_iri": r["investment_iri"],
            "fund_name": r["fund_name_kr"],
            "gp_name": r["gp_name_kr"],
            "n_branches": r["n_branches"],
        }
        for r in instance_records
    ]


# ============================================
# 입력 검증
# ============================================

def validate_input(fund_name_kr, gp_name_kr, branches_data):
    """
    사용자 입력 검증. 빈 값, 너무 긴 값, 부적절한 문자 등 체크.
    
    Returns:
        (is_valid, error_message)
    """
    if not fund_name_kr or not fund_name_kr.strip():
        return False, "펀드명을 입력해주세요."
    
    if not gp_name_kr or not gp_name_kr.strip():
        return False, "GP명을 입력해주세요."
    
    if len(fund_name_kr) > 50:
        return False, "펀드명은 50자 이하로 입력해주세요."
    
    if len(gp_name_kr) > 50:
        return False, "GP명은 50자 이하로 입력해주세요."
    
    # XML/HTML 태그 차단
    if re.search(r'<[^>]+>', fund_name_kr) or re.search(r'<[^>]+>', gp_name_kr):
        return False, "이름에 HTML 태그를 포함할 수 없습니다."
    
    if not branches_data or len(branches_data) == 0:
        return False, "최소 1개의 분기를 등록해주세요."
    
    if len(branches_data) > 3:
        return False, "분기는 최대 3개까지 등록 가능합니다."
    
    for idx, branch in enumerate(branches_data, start=1):
        if not branch.get("product_iri"):
            return False, f"분기 {idx}의 상품 유형을 선택해주세요."
        if not branch.get("stage_iri"):
            return False, f"분기 {idx}의 단계를 선택해주세요."
        if not branch.get("state_iri"):
            return False, f"분기 {idx}의 상태를 선택해주세요."
        try:
            amt = int(branch.get("amount", 0))
            if amt <= 0 or amt > 100000:
                return False, f"분기 {idx}의 금액은 1억~10만억 사이여야 합니다."
        except (ValueError, TypeError):
            return False, f"분기 {idx}의 금액이 올바르지 않습니다."
    
    return True, ""
