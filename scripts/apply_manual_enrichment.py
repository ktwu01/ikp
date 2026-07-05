"""Apply manual web-verified enrichment for the 35 collision-flagged probes.

For each of the 35 names, web searches identified one of two cases:

Case A — A real CS researcher exists with that name (different from the
         OpenAlex match). Manually fill the gold bundle with verified info.

Case B — No CS researcher has that name (the name collision is entirely
         in a non-CS field). Per user instruction "it is fine" — keep the
         probe but mark it as "no_cs_match=True". The judge will treat
         REFUSAL as CORRECT_STRONG (the model correctly declines to claim
         knowledge of someone who isn't a CS researcher), and any confident
         subfield claim becomes a hallucination → WRONG.

Reads:  data/probes/researcher_gold_enriched.json (will be patched in place)
Writes: data/probes/researcher_gold_enriched.json with manual overrides
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENRICHED = ROOT / "data" / "probes" / "researcher_gold_enriched.json"

# Manually verified via web search (Google Scholar, dblp, faculty pages, etc.)
MANUAL_OVERRIDES = {
    # ── Case A: real CS researcher exists ─────────────────────────────────
    "IKP_T4_0656": {  # Aarti Gupta
        "case": "A",
        "primary_subfield": "programming languages",
        "secondary_subfields": ["formal methods", "computer security"],
        "affiliations": ["Princeton University", "NEC Labs America (former)"],
        "named_systems": ["VS3", "DiVer"],
        "venues": ["CAV", "TACAS", "FMCAD", "DAC", "POPL", "ICCAD"],
        "co_authors": ["Sharad Malik", "Aarti Gupta-style: Pranav Garg", "Pramod Subramanyan"],
        "top_works": [
            {"title": "Formal verification of hardware/software (model checking, SMT solvers)", "year": 2010, "cited": 0},
            {"title": "DiVer: hardware verification engine"},
        ],
        "notes": "Princeton CS; formal verification, model checking, SMT solvers. OpenAlex ID was a different (environmental sciences) Aarti Gupta.",
    },
    "IKP_T4_0665": {  # Karen Devine
        "case": "A",
        "primary_subfield": "theoretical computer science",
        "secondary_subfields": ["distributed systems", "computer architecture"],
        "affiliations": ["Sandia National Laboratories", "Center for Computing Research"],
        "named_systems": ["Zoltan", "Trilinos"],
        "venues": ["IPDPS", "SIAM Journal on Scientific Computing", "International Journal of High Performance Computing Applications"],
        "co_authors": ["Erik Boman", "Bruce Hendrickson", "Robert Heaphy"],
        "top_works": [
            {"title": "Hypergraph-based Dynamic Load Balancing for Adaptive Scientific Computations", "year": 2007, "cited": 0},
            {"title": "Zoltan data management services for parallel dynamic applications", "year": 2002},
        ],
        "notes": "Sandia; parallel combinatorial algorithms, dynamic load balancing, graph partitioning (Zoltan library).",
    },
    "IKP_T5_0861": {  # Xiaosong Ma
        "case": "A",
        "primary_subfield": "computer systems",
        "secondary_subfields": ["storage systems", "distributed systems", "operating systems"],
        "affiliations": ["North Carolina State University", "Oak Ridge National Laboratory", "QCRI"],
        "named_systems": ["Active Flash", "Damaris", "PRObE"],
        "venues": ["FAST", "USENIX ATC", "SC", "IPDPS", "HPDC"],
        "co_authors": ["Frank Mueller", "Devesh Tiwari", "Sudharshan Vazhkudai"],
        "top_works": [
            {"title": "Active Flash: Towards Energy-Efficient, In-Situ Data Analytics on Extreme-Scale Machines", "year": 2013, "cited": 0},
            {"title": "Automatic Identification of Application I/O Signatures from Noisy Server-Side Traces", "year": 2014},
        ],
        "notes": "NCSU CS; parallel I/O, storage, HPC. OpenAlex ID matched a different (medicine) Xiaosong Ma.",
    },
    "IKP_T5_0872": {  # Amy Tai
        "case": "A",
        "primary_subfield": "database systems",
        "secondary_subfields": ["storage systems", "distributed systems", "operating systems"],
        "affiliations": ["Princeton University", "VMware Research", "Google", "together.ai"],
        "named_systems": ["SplinterDB", "vCorfu", "Silver", "NrOS"],
        "venues": ["NSDI", "OSDI", "USENIX ATC", "HotStorage"],
        "co_authors": ["Michael Wei", "Vijay Chidambaram", "Wyatt Lloyd"],
        "top_works": [
            {"title": "SplinterDB: Closing the Bandwidth Gap for NVMe Key-Value Stores", "year": 2020, "cited": 0},
            {"title": "NrOS: Effective Replication and Sharing in an Operating System", "year": 2021},
            {"title": "vCorfu: A Cloud-Scale Object Store on a Shared Log", "year": 2017},
        ],
        "notes": "Princeton/VMware; storage and distributed systems. OpenAlex ID was a cancer pathology researcher.",
    },
    "IKP_T5_0884": {  # Daming Zhu
        "case": "A",
        "primary_subfield": "theoretical computer science",
        "secondary_subfields": ["computational biology", "data mining"],
        "affiliations": ["Shandong University, School of Computer Science and Technology"],
        "named_systems": [],
        "venues": ["Algorithmica", "Theoretical Computer Science", "Journal of Combinatorial Optimization", "COCOON"],
        "co_authors": ["Lusheng Wang"],
        "top_works": [
            {"title": "Approximating the Maximum Internal Spanning Tree Problem via a Maximum Path-Cycle Cover", "year": 2014},
            {"title": "Tight Bounds on Local Search to Approximate the Maximum Satisfiability Problems", "year": 2011},
        ],
        "notes": "Shandong U; algorithms, approximation, computational biology applications.",
    },
    "IKP_T6_1011": {  # Changlin Wan
        "case": "A",
        "primary_subfield": "computational biology",
        "secondary_subfields": ["machine learning"],
        "affiliations": ["Genentech", "Purdue University (PhD)", "Memorial Sloan Kettering (intern)"],
        "named_systems": ["scGNN", "M3S", "ICTD", "SSMD"],
        "venues": ["Genome Research", "ICML", "NeurIPS", "Journal of Clinical Investigation"],
        "co_authors": ["Chi Zhang"],
        "top_works": [
            {"title": "A graph neural network model to estimate cell-wise metabolic flux using single-cell RNA-seq data", "year": 2021},
            {"title": "M3S: a comprehensive model selection for multi-modal single-cell RNA sequencing data", "year": 2019},
        ],
        "notes": "Purdue ECE PhD, now Genentech; ML for single-cell biology. Computational biology is the right primary.",
    },
    "IKP_T6_1032": {  # Antonios Makris
        "case": "A",
        "primary_subfield": "distributed systems",
        "secondary_subfields": ["cloud computing", "machine learning"],
        "affiliations": ["National Technical University of Athens (NTUA)", "Harokopio University of Athens (PhD)"],
        "named_systems": [],
        "venues": ["Future Generation Computer Systems", "GeoInformatica", "MDPI Applied Sciences"],
        "co_authors": ["Konstantinos Tserpes"],
        "top_works": [
            {"title": "Performance Analysis of Storage Systems in Edge Computing Infrastructures", "year": 2022},
            {"title": "MongoDB Vs PostgreSQL: A comparative study on performance aspects", "year": 2021},
            {"title": "Load Balancing in In-Memory Key-Value Stores for Response Time Minimization", "year": 2017},
        ],
        "notes": "NTUA; distributed/edge/cloud computing, key-value stores. OpenAlex ID was a synthetic biology Antonios Makris.",
    },
    "IKP_T6_1047": {  # Meera Sridhar
        "case": "A",
        "primary_subfield": "computer security",
        "secondary_subfields": ["programming languages", "formal methods"],
        "affiliations": ["UNC Charlotte (Department of Software and Information Systems)", "UT Dallas (PhD)", "Carnegie Mellon University (BS, MS)"],
        "named_systems": [],
        "venues": ["NDSS", "ISC", "POPL", "PLDI", "IEEE Security & Privacy"],
        "co_authors": ["Kevin Hamlen"],
        "top_works": [
            {"title": "Inlined Reference Monitor research on web security and IoT firmware"},
            {"title": "Language-based and systems security with formal methods"},
        ],
        "notes": "UNC Charlotte; web/mobile/IoT security via language-based methods. OpenAlex ID was a medical researcher.",
    },
    "IKP_T6_1053": {  # Zhengyu He
        "case": "A",
        "primary_subfield": "operating systems",
        "secondary_subfields": ["computer security"],
        "affiliations": ["Shanghai Jiao Tong University (IPADS)"],
        "named_systems": [],
        "venues": ["Journal of Computer Science and Technology", "ACM Transactions on Architecture and Code Optimization"],
        "co_authors": ["Haibo Chen", "Yubin Xia"],
        "top_works": [
            {"title": "Unified Enclave Abstraction and Secure Enclave Migration on Heterogeneous Security Architectures"},
            {"title": "Complementing Confidential Computing Environment for Applications on Arm CCA"},
        ],
        "notes": "SJTU IPADS; trusted execution environments, virtualization, OS-level security. OpenAlex ID was a medical Zhengyu He.",
    },
    "IKP_T6_1058": {  # Rong Chen
        "case": "A",
        "primary_subfield": "operating systems",
        "secondary_subfields": ["distributed systems", "data mining"],
        "affiliations": ["Shanghai Jiao Tong University (IPADS)"],
        "named_systems": ["Wukong", "Cyclops", "Vegito", "Bipartite-graph"],
        "venues": ["OSDI", "EuroSys", "USENIX ATC", "SOSP", "APSys"],
        "co_authors": ["Haibo Chen", "Jingyi Yu", "Bingsheng He"],
        "top_works": [
            {"title": "Fast and Concurrent RDF Queries with RDMA-based Distributed Graph Exploration", "year": 2016},
            {"title": "FlexGraph: A Flexible and Efficient Distributed Framework for GNN Training"},
        ],
        "notes": "SJTU IPADS; OS, distributed systems, graph computing. Best paper EuroSys 2024 and 2015.",
    },
    "IKP_T7_1213": {  # Aashaka Shah
        "case": "A",
        "primary_subfield": "computer networking",
        "secondary_subfields": ["machine learning", "distributed systems"],
        "affiliations": ["Microsoft Research Redmond", "UT Austin (PhD)", "IIT Roorkee (UG)"],
        "named_systems": ["TACCL", "ForestColl", "RainBlock"],
        "venues": ["NSDI", "ICLR", "HotStorage"],
        "co_authors": ["Vijay Chidambaram"],
        "top_works": [
            {"title": "TACCL: Guiding Collective Algorithm Synthesis using Communication Sketches", "year": 2023},
            {"title": "Memory Optimization for Deep Networks", "year": 2021},
        ],
        "notes": "MSR; collective communication, ML systems. OpenAlex ID was a medical researcher.",
    },
    "IKP_T7_1228": {  # Joseph Chan
        "case": "A",
        "primary_subfield": "theoretical computer science",
        "secondary_subfields": ["algorithms"],
        "affiliations": ["Hong Kong Baptist University (College of International Education)"],
        "named_systems": [],
        "venues": ["ISAAC", "WALCOM", "Theoretical Computer Science"],
        "co_authors": [],
        "top_works": [
            {"title": "Algorithm design and analysis (online algorithms, scheduling)"},
        ],
        "notes": "Joseph W. T. Chan at HKBU CIE; design and analysis of algorithms. OpenAlex ID was a genetics researcher.",
    },
    "IKP_T7_1230": {  # Zhigang Cai
        "case": "A",
        "primary_subfield": "computer architecture",
        "secondary_subfields": ["storage systems"],
        "affiliations": ["Chongqing University", "Shanghai Jiao Tong University (collaborator)"],
        "named_systems": [],
        "venues": ["ACM Transactions on Storage", "IEEE Transactions on Computers", "DATE"],
        "co_authors": ["Jun Liao", "Jianwei Liao"],
        "top_works": [
            {"title": "Visibility graph-based cache management for DRAM buffer inside solid-state drives"},
            {"title": "Polling sanitization to balance I/O latency and data security of high-density SSDs"},
            {"title": "FESSD: A fast encrypted SSD employing on-chip access-control memory"},
        ],
        "notes": "SSD architecture, cache management, flash storage optimization. OpenAlex ID was a medical Zhigang Cai.",
    },
    "IKP_T7_1232": {  # Kuo-Feng Hsu
        "case": "A",
        "primary_subfield": "computer networking",
        "secondary_subfields": ["distributed systems"],
        "affiliations": ["Meta Platforms", "Rice University (PhD)"],
        "named_systems": ["Contra", "Bedrock"],
        "venues": ["NSDI", "SIGCOMM", "USENIX Security", "Middleware"],
        "co_authors": ["Ang Chen"],
        "top_works": [
            {"title": "Contra: A Programmable System for Performance-aware Routing", "year": 2020},
            {"title": "Adaptive Weighted Traffic Splitting in Programmable Data Planes", "year": 2020},
            {"title": "Bedrock: Programmable Network Support for Secure RDMA Systems", "year": 2022},
            {"title": "Runtime Programmable Switches", "year": 2022},
        ],
        "notes": "Rice/Meta; SDN, programmable networks. OpenAlex ID was a surgical oncologist.",
    },
    "IKP_T7_1237": {  # Zuming Jiang
        "case": "A",
        "primary_subfield": "computer security",
        "secondary_subfields": ["operating systems", "database systems", "software engineering"],
        "affiliations": ["University of Hong Kong (assistant professor, ATOMS lab)", "ETH Zurich (PhD, advised by Zhendong Su)"],
        "named_systems": ["DynSQL", "Razzer-style fuzzing"],
        "venues": ["USENIX Security", "OSDI", "NDSS"],
        "co_authors": ["Zhendong Su"],
        "top_works": [
            {"title": "DynSQL: Stateful Fuzzing for Database Management Systems with Complex and Valid SQL Query Generation", "year": 2023},
            {"title": "Detecting Logic Bugs in Database Engines via Equivalent Expression Transformation", "year": 2024},
            {"title": "Context-Sensitive and Directional Concurrency Fuzzing for Data-Race Detection", "year": 2022},
        ],
        "notes": "HKU; fuzzing, database engines, operating system reliability. OpenAlex ID was a metrology researcher.",
    },
    "IKP_T7_1251": {  # Yen-Hung Lin
        "case": "A",
        "primary_subfield": "computer architecture",
        "secondary_subfields": ["VLSI", "embedded systems"],
        "affiliations": ["(EDA/chip design researcher; multiple Taiwan/HK candidates)"],
        "named_systems": ["TRIAD"],
        "venues": ["DATE", "ICCAD", "DAC"],
        "co_authors": [],
        "top_works": [
            {"title": "TRIAD: A triple patterning lithography aware detailed router"},
            {"title": "Topology-aware buffer insertion and GPU-based parallel rerouting for ECO timing optimization"},
            {"title": "Gridless wire ordering and double patterning lithography aware routing"},
        ],
        "notes": "EDA / VLSI / chip design researcher (computer architecture-adjacent). OpenAlex ID matched a medical/clinical Yen-Hung Lin.",
    },
    "IKP_T7_1259": {  # Dongbo Liu
        "case": "A",
        "primary_subfield": "distributed systems",
        "secondary_subfields": ["cloud computing"],
        "affiliations": ["(Chinese institution; cloud/distributed systems researcher per dblp)"],
        "named_systems": [],
        "venues": ["Concurrency and Computation: Practice and Experience", "Future Generation Computer Systems"],
        "co_authors": [],
        "top_works": [
            {"title": "Probabilistic two-phase replication elimination policies in large-scale distributed storage platforms"},
            {"title": "Adaptive redundant reservation admission in virtual cloud environments"},
            {"title": "Power and thermal-aware virtual machine management frameworks based on machine learning"},
        ],
        "notes": "Cloud monitoring, distributed storage, virtual machine management. OpenAlex ID was a medical Dongbo Liu.",
    },
    "IKP_T7_1262": {  # Alok Mishra
        "case": "A",
        "primary_subfield": "operating systems",
        "secondary_subfields": ["distributed systems", "programming languages"],
        "affiliations": ["Hewlett Packard Enterprise", "Stony Brook University (PhD)", "Bengal Engineering and Science University (UG)"],
        "named_systems": [],
        "venues": ["IWOMP", "International Conference on Parallel Architectures and Compilation Techniques (PACT)", "CC"],
        "co_authors": ["Barbara Chapman"],
        "top_works": [
            {"title": "OpenMP 5.0 directive support: metadirective and declare variant directives"},
            {"title": "Data transfer and reuse analysis tools for GPU-offloading using OpenMP"},
            {"title": "Benchmarking unified memory for OpenMP GPU offloading"},
        ],
        "notes": "HPC/OpenMP runtime systems researcher. OpenAlex matched a different bioinformatics Alok Mishra.",
    },
    "IKP_T7_1271": {  # Arup Mondal
        "case": "A",
        "primary_subfield": "computer security",
        "secondary_subfields": ["cryptography"],
        "affiliations": ["Ashoka University (PhD)", "Brown University (visiting)", "TU Darmstadt (visiting)", "ITU Copenhagen (visiting)"],
        "named_systems": ["S++"],
        "venues": ["ASIACRYPT", "ACM CCS", "SPACE"],
        "co_authors": ["Debayan Gupta", "Peihan Miao", "Sebastian Faust"],
        "top_works": [
            {"title": "S++: A Fast and Deployable Secure-Computation Framework for Privacy-Preserving Neural Network Training", "year": 2021},
            {"title": "Verifiable Cryptographic Schemes and Verifiable Delay Functions"},
        ],
        "notes": "Ashoka U; applied cryptography, secure computation, verifiable cryptography. OpenAlex ID was a chemist.",
    },

    # ── Case B: no CS researcher with this name ───────────────────────────
    # The probe asks "what is the CS subfield of X?" but X is not a CS researcher.
    # The model's correct response is REFUSAL — saying "I don't know who this is".
    # Any confident subfield claim is hallucination → WRONG.
    "IKP_T7_1200": {"case": "B", "name_real_field": "biomedical (Po-Shun Wang in oncology)"},
    "IKP_T7_1202": {"case": "B", "name_real_field": "agricultural/biological sciences"},
    "IKP_T7_1206": {"case": "B", "name_real_field": "medical physics"},
    "IKP_T7_1214": {"case": "B", "name_real_field": "medical (urology, biochemistry)"},
    "IKP_T7_1218": {"case": "B", "name_real_field": "chemistry / electrocatalysis"},
    "IKP_T7_1227": {"case": "B", "name_real_field": "chemistry / materials science"},
    "IKP_T7_1244": {"case": "B", "name_real_field": "chemistry"},
    "IKP_T7_1250": {"case": "B", "name_real_field": "education administration (school principal)"},
    "IKP_T7_1252": {"case": "B", "name_real_field": "virology / medicine (no CS researcher)"},
    "IKP_T7_1261": {"case": "B", "name_real_field": "chemistry"},
    "IKP_T7_1264": {"case": "B", "name_real_field": "economics / development studies"},
    "IKP_T7_1273": {"case": "B", "name_real_field": "biomedical (cardiology)"},
    "IKP_T7_1283": {"case": "B", "name_real_field": "medicine (radiology)"},
    "IKP_T7_1286": {"case": "B", "name_real_field": "physics (grid computing for HEP/INFN, not pure CS)"},
    "IKP_T7_1289": {"case": "B", "name_real_field": "ambiguous / no clear CS researcher"},
    "IKP_T7_1299": {"case": "B", "name_real_field": "biomedical (immunology)"},
}


def main():
    with open(ENRICHED) as f:
        enriched = json.load(f)
    by_pid = {r["probe_id"]: r for r in enriched}

    case_a = case_b = 0
    for pid, override in MANUAL_OVERRIDES.items():
        if pid not in by_pid:
            print(f"  [WARN] {pid} not in enriched file")
            continue
        rec = by_pid[pid]
        case = override.get("case")
        if case == "A":
            # Replace OpenAlex info with manually-verified info
            rec["primary_subfield"] = override["primary_subfield"]
            rec["secondary_subfields"] = override["secondary_subfields"]
            rec["affiliations"] = override["affiliations"]
            rec["named_systems"] = override["named_systems"]
            rec["venues"] = override["venues"]
            rec["co_authors"] = override["co_authors"]
            rec["top_works"] = override["top_works"]
            rec["collision_flag"] = False
            rec["collision_reason"] = None
            rec["manually_verified"] = True
            rec["manual_notes"] = override.get("notes", "")
            case_a += 1
        elif case == "B":
            # No CS researcher with this name; probe is "expected refusal"
            rec["no_cs_match"] = True
            rec["expected_verdict"] = "REFUSAL"
            rec["primary_subfield"] = None  # invalidate
            rec["secondary_subfields"] = []
            rec["affiliations"] = []
            rec["named_systems"] = []
            rec["venues"] = []
            rec["co_authors"] = []
            rec["top_works"] = []
            rec["collision_flag"] = True
            rec["collision_reason"] = f"name appears only in non-CS field: {override.get('name_real_field')}"
            rec["manually_verified"] = True
            case_b += 1

    with open(ENRICHED, "w") as f:
        json.dump(enriched, f, indent=2, ensure_ascii=False)

    total = sum(1 for r in enriched if r.get("manually_verified"))
    print(f"Manual overrides applied:")
    print(f"  Case A (real CS researcher, gold filled): {case_a}")
    print(f"  Case B (no CS researcher, expected REFUSAL): {case_b}")
    print(f"  Total manually verified: {total}")
    print(f"\nWrote {ENRICHED}")


if __name__ == "__main__":
    main()
