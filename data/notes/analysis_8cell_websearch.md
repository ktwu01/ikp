# 8-Cell Web Presence Analysis: Why Some Researchers Are Recognized by LLMs

## Methodology

For each of 20 researchers, we searched the web to characterize their total digital footprint:
personal websites, Wikipedia pages, GitHub projects, blog posts by others, news coverage,
tutorials, social media, etc. The goal is to understand what predicts LLM recognition
beyond raw citation count.

---

## Cell 1: Systems + High Citations + Well-Recognized

### Thorsten Joachims (52K citations, 97% recognition, information retrieval)

| Dimension | Finding |
|-----------|---------|
| **Personal site** | Yes -- cs.cornell.edu/people/tj/ |
| **Wikipedia** | No dedicated page |
| **Google Scholar** | 88K citations (higher than OpenAlex) |
| **Open-source software** | SVMlight (one of the most widely-used SVM implementations in the 2000s), SVMrank, SVMstruct -- all hosted at svmlight.joachims.org |
| **GitHub presence** | Multiple third-party wrappers (pysvmlight, Java JNI, MATLAB interface) |
| **Secondary content** | Tutorials by others on using SVMlight; his "learning to rank" papers are cited in IR textbooks; Stanford CS276 lecture slides reference his work |
| **News/media** | Cornell Chronicle (Jan 2026): named Vice Provost for AI Strategy; Cornell Daily Sun coverage |
| **Conference presence** | Major figure in SIGIR/KDD/ICML communities |
| **Name uniqueness** | Highly unique -- "Thorsten Joachims" returns only this person |
| **Key recognition driver** | SVMlight was ubiquitous software in the 2000s-2010s; "learning to rank" is a named subfield he helped create; textbook-level impact |

### Mario Gerla (54K citations, 89% recognition, computer networking)

| Dimension | Finding |
|-----------|---------|
| **Personal site** | UCLA Network Research Lab page |
| **Wikipedia** | YES -- full Wikipedia article |
| **Google Scholar** | Heavily cited; 1000+ papers |
| **Open-source software** | No widely-known open-source tools |
| **Secondary content** | Obituaries in UCLA, IMDEA Networks, IEEE; best paper award named after him (MedComNet); ISSNAF Young Investigator Award named after him |
| **News/media** | LA Times obituary; UCLA Newsroom; extensive memorial coverage |
| **Historical significance** | ARPANET pioneer -- helped design the topology of the predecessor to the internet under Leonard Kleinrock at UCLA |
| **Conference presence** | IEEE Fellow; IEEE INFOCOM Achievement Award (2018); Postel Chair in Networking |
| **Name uniqueness** | Fairly unique -- "Mario Gerla" overwhelmingly returns this person |
| **Key recognition driver** | Wikipedia page; ARPANET connection; IEEE Fellow; 100+ PhD graduates; named awards |

**Cell 1 Summary:** Both researchers have massive secondary web footprints. Joachims has widely-used software; Gerla has a Wikipedia page and historical ARPANET significance. Both names are unique.

---

## Cell 2: Systems + High Citations + NOT Recognized

### CRITICAL FINDING: Both "high citation" researchers in this cell are OpenAlex name collisions.

### Yan Jiao (OpenAlex says 46K citations; 6% recognition, T7, "computer networking")

| Dimension | Finding |
|-----------|---------|
| **OpenAlex ID** | A5034195419 |
| **OpenAlex actual person** | Professor Yan Jiao, University of Adelaide, CHEMISTRY (electrocatalysts, CO2 reduction, energy materials) -- NOT a computer networking researcher |
| **Actual citations** | The 46K citations belong to the Adelaide chemistry professor, a Clarivate Highly Cited Researcher since 2019 |
| **The real CS "Yan Jiao"** | Likely a different, much less-cited researcher. Google Scholar shows a "Yan Jiao" at Google with ~1,282 citations in ML/RL/bioinformatics |
| **Web presence of CS Yan Jiao** | Minimal -- no personal website, no Wikipedia, no open-source projects found |
| **Name uniqueness** | VERY COMMON Chinese name; multiple "Yan Jiao" researchers across fields |
| **Diagnosis** | OpenAlex merged a chemistry professor's profile with a CS researcher's name. The 46K citations are entirely from chemistry papers. The actual CS "Yan Jiao" likely has <2K citations. |

### Xinming Wang (OpenAlex says 44K citations; 1% recognition, T7, "HCI")

| Dimension | Finding |
|-----------|---------|
| **OpenAlex ID** | A5100643640 |
| **OpenAlex actual person** | Professor Xinming Wang, Guangzhou Institute of Geochemistry, Chinese Academy of Sciences -- ATMOSPHERIC CHEMISTRY (volatile organic compounds, organic aerosols, smog chamber studies) |
| **Actual citations** | The 44K citations (h-index 102) belong to the environmental chemist with 1,434 works |
| **The real CS "Xinming Wang"** | A different person who works on gaze estimation, fatigue detection, action recognition. DBLP shows an unresolved disambiguation page. Much lower citations. |
| **Web presence of CS Xinming Wang** | Minimal -- no personal website, no Wikipedia, no open-source projects |
| **Name uniqueness** | EXTREMELY COMMON Chinese name; "Xinming Wang" returns the chemist first, then multiple unrelated people |
| **Diagnosis** | Same pattern as Yan Jiao -- OpenAlex merged an atmospheric chemist's profile with a CS researcher sharing the same name. The actual CS researcher likely has <500 citations. |

### Sylvie Dujardin (149 citations, 0% recognition, T7, "operating systems")

**Also a misclassification**, though less impactful:

| Dimension | Finding |
|-----------|---------|
| **OpenAlex ID** | A5048526688 |
| **OpenAlex actual person** | A sleep/neuroscience researcher at Kempenhaeghe (Netherlands) and Centre Hospitalier Universitaire Brugmann (Belgium). 9 works, all on sleep disorders and circadian rhythms. |
| **Actual field** | Medicine/neuroscience -- NOT operating systems |
| **Web presence** | ResearchGate profile (as neurologist); no CS presence whatsoever |
| **Name uniqueness** | Moderately unique in French; only one prominent "Sylvie Dujardin" found (the neurologist) |
| **Diagnosis** | OpenAlex tagged "operating systems" as a concept for this author, likely an artifact. There is no CS researcher named Sylvie Dujardin. |

**Cell 2 Summary:** ALL THREE "unrecognized high-citation" researchers are OpenAlex disambiguation errors or field misclassifications. The high citations belong to same-named researchers in chemistry, environmental science, and neuroscience. This is a critical data quality finding -- these should not be used as evidence that "high-citation researchers can go unrecognized." The actual CS researchers with these names (if they exist) have far fewer citations.

---

## Cell 3: Systems + Low Citations + Well-Recognized

### Yiannis Psaras (318 citations, 69% recognition, T3, distributed systems)

| Dimension | Finding |
|-----------|---------|
| **Personal site** | Yes -- Protocol Labs Research profile, ProbeLab profile |
| **Wikipedia** | No, but has a Wikidata entry (Q23299672) |
| **Google Scholar** | ~318 citations (modest) |
| **Industry role** | Research Scientist at Protocol Labs, working on IPFS and libp2p |
| **Open-source/web3** | Deeply involved in IPFS (InterPlanetary File System) -- one of the largest decentralized P2P networks. Work on libp2p, Filecoin, drand |
| **Secondary content** | IPFS blog posts mentioning him; Protocol Labs blog posts; FOSDEM 2021 speaker; IEEE Transmitter author |
| **Conferences** | EPSRC Fellow (UK); UCL lecturer; 5 Best Paper Awards |
| **Name uniqueness** | Highly unique -- "Yiannis Psaras" returns only this person |
| **Key recognition driver** | IPFS/Protocol Labs ecosystem is extremely well-known in tech. Crypto/web3 communities generate enormous secondary content. The IPFS connection creates a massive web footprint despite modest citations. |

### Kristopher Micinski (850 citations, 59% recognition, T3, programming languages)

| Dimension | Finding |
|-----------|---------|
| **Personal site** | Yes -- kmicinski.com with blog and teaching materials |
| **Wikipedia** | No |
| **Google Scholar** | ~850 citations |
| **YouTube** | FREE course lectures on YouTube (CIS352 Principles of Programming Languages, Spring 2022, 22+ lectures) + Racket tutorial videos |
| **GitHub** | Active -- personal repos for program analysis, compiler courses, OCaml examples |
| **Blog** | Active research blog at kmicinski.com/blog/ |
| **Teaching materials** | Extensive open course materials; "Build a Compiler in Five Projects" blog post |
| **Conference presence** | ICFP 2024, POPL 2023, SPLASH 2024, CC 2021 |
| **Name uniqueness** | Very unique -- "Kristopher Micinski" returns only this person |
| **Key recognition driver** | YouTube lectures + open teaching materials + active blog. The PL community is tight-knit and values educational content. Free course lectures on YouTube expose his name to many students/practitioners. |

### Susan Lysecky (382 citations, 54% recognition, T3, embedded systems)

| Dimension | Finding |
|-----------|---------|
| **Personal site** | No dedicated personal site |
| **Wikipedia** | No |
| **Google Scholar** | ~382 citations |
| **zyBooks** | Senior Content Developer at zyBooks -- widely-used interactive STEM textbooks |
| **Textbook impact** | zyBooks are used in hundreds of university courses; her name appears as author/developer in CS course syllabi nationwide |
| **Rate My Professors** | Has a profile -- students know her name from University of Arizona (2006-2014) |
| **Research** | eBlocks -- embedded systems for non-programmers |
| **Name uniqueness** | Fairly unique -- "Susan Lysecky" returns primarily this person |
| **Key recognition driver** | zyBooks textbooks are used by hundreds of thousands of CS students. Even if students don't cite her papers, they encounter her name in course materials. Textbook authorship creates enormous name exposure in training data. |

**Cell 3 Summary:** All three are recognized despite low citations because they have non-academic web presence that generates training data:
- Psaras: IPFS/web3 ecosystem (massive community blog/docs footprint)
- Micinski: YouTube lectures + blog (educational content indexed by crawlers)
- Lysecky: zyBooks textbooks (appear in thousands of course syllabi and university websites)

**Key insight:** Non-academic web presence (industry tools, educational content, web3 communities) can drive LLM recognition more effectively than citations alone.

---

## Cell 4: Systems + Low Citations + NOT Recognized

### Sylvie Dujardin (149 citations, 0% recognition, T7, "operating systems")

(See Cell 2 analysis -- this is actually a sleep researcher, not a CS researcher at all.)

### Qiansheng Rao (15 citations, 0% recognition, T7, computer architecture)

| Dimension | Finding |
|-----------|---------|
| **Personal site** | None found |
| **Wikipedia** | No |
| **Google Scholar** | Not found as a CS researcher |
| **DBLP** | Not found |
| **Web presence** | Near zero. One co-authored paper found on semiconductor IGBTs (microelectronics). |
| **OpenAlex** | 11 works, 15 citations, h-index 2 |
| **Secondary content** | None |
| **Name uniqueness** | Moderately common Chinese name, but no prominent researcher with this name in any field |
| **Key non-recognition driver** | Essentially no web footprint. 11 papers with 15 total citations. No personal site, no open source, no teaching materials, no blog, no media coverage. |

**Cell 4 Summary:** Qiansheng Rao is a textbook case of minimal web presence -- a researcher whose work exists only in a handful of papers with negligible citation impact. Sylvie Dujardin is a data error (not a CS researcher). The finding holds: researchers with <20 citations and no non-academic web presence are invisible to LLMs.

---

## Cell 5: ML + High Citations + Well-Recognized (100% or near)

### Tri Dao (2,817 citations [OpenAlex], ~15K [GScholar], 100% recognition)

| Dimension | Finding |
|-----------|---------|
| **Personal site** | Yes -- tridao.me with blog posts |
| **Wikipedia** | No personal page, but FlashAttention is widely discussed; Mamba has a Wikipedia article |
| **Google Scholar** | ~15,338 citations |
| **GitHub** | Dao-AILab/flash-attention: **23.4K stars**, 2.6K forks. state-spaces/mamba: **18K stars** |
| **Blog** | Active blog at tridao.me (FlashAttention-3, Mamba-2 posts) |
| **Secondary content by others** | MASSIVE: Visual Guide to Mamba (50+ custom visuals); Medium/Towards Data Science/Towards AI explanations; HuggingFace integration guides; YouTube tutorials; IBM explainer; Together.ai blog; Imbue podcast; TED AI speaker |
| **News/media** | Together.ai Chief Scientist; Schmidt AI2050 Fellow; Princeton incoming faculty |
| **Industry adoption** | FlashAttention used by virtually every major LLM (LLaMA, Falcon, MPT, RedPajama, etc.) |
| **Name uniqueness** | Fairly unique -- "Tri Dao" returns primarily this person |
| **Key recognition driver** | FlashAttention is perhaps the single most impactful systems contribution to the LLM era. 23K GitHub stars. Every LLM practitioner knows this name. Mamba added a second viral project. |

### Albert Gu (3,015 citations [OpenAlex], 86% recognition)

| Dimension | Finding |
|-----------|---------|
| **Personal site** | CMU faculty page; goombalab.github.io |
| **Wikipedia** | Mamba (deep learning architecture) has a Wikipedia article mentioning him by name |
| **Google Scholar** | ~3,015 citations |
| **GitHub** | state-spaces/mamba: 18K stars; state-spaces/s4 also highly starred |
| **Secondary content by others** | Extensive: Visual Guide to Mamba and State Space Models; Towards Data Science articles; Medium explainers; "Awesome Mamba" paper lists on GitHub; Tower Research interview |
| **News/media** | TIME 100 Most Influential People in AI 2024; CMU news; Cartesia co-founder |
| **Industry** | Co-founded Cartesia ($100M raised); Chief Scientist |
| **Name uniqueness** | Moderately common name, but "Albert Gu" + ML/AI context returns only this person |
| **Key recognition driver** | TIME 100 AI list + Mamba Wikipedia article + Cartesia startup + 18K GitHub stars. S4/Mamba defined a new subfield (state space models). |

### Aditi Raghunathan (4,689 citations, 100% recognition)

| Dimension | Finding |
|-----------|---------|
| **Personal site** | Yes -- cs.cmu.edu/~aditirag/ and stanford.edu/~aditir/ |
| **Wikipedia** | No |
| **Google Scholar** | ~4,689 citations |
| **GitHub** | p-lambda/robust_tradeoff and related repos |
| **Secondary content** | Academic talks widely referenced; Schmidt AI2050 Community Perspective essay; DeepAI profile |
| **Awards** | Schmidt AI2050 Fellowship; Stanford Arthur Samuel Best Thesis Award; Google PhD Fellowship; Open Philanthropy AI Fellowship |
| **Invited talks** | MIT EECS seminar; UChicago CS seminar; ICML 2023 tutorial |
| **News/media** | CMU CyLab profile; Carnegie Bosch Fellowship |
| **Name uniqueness** | Highly unique -- "Aditi Raghunathan" returns only this person |
| **Key recognition driver** | Unique name + numerous prestigious fellowships (each generates web content) + highly active in the robustness/safety community which is heavily discussed in AI discourse. Her work on adversarial robustness connects to the hot topic of AI safety. |

**Cell 5 Summary:** All three have enormous secondary web footprints. Tri Dao and Albert Gu have the most GitHub stars of anyone in this study (23K and 18K). Aditi Raghunathan benefits from a perfectly unique name and numerous fellowship announcements. All three work in areas (efficient attention, SSMs, robustness/safety) that are central to current AI discourse.

---

## Cell 6: ML + High Citations + Lower Recognition (57%)

### Eric Mitchell (GScholar: ~15K citations; OpenAlex: 74K [NAME COLLISION]; 57% recognition)

| Dimension | Finding |
|-----------|---------|
| **Personal site** | Yes -- ericmitchell.ai |
| **Wikipedia** | "Eric Mitchell" Wikipedia page exists but is about a film director/actor |
| **Google Scholar** | ~15,338 citations (correct figure) |
| **OpenAlex issue** | OpenAlex reports 74K citations, h-index 135 -- this is a NAME COLLISION. OpenAlex merged multiple "Eric Mitchell" profiles. The real ML researcher has ~15K citations. |
| **GitHub** | eric-mitchell/direct-preference-optimization (DPO reference implementation); eric-mitchell/detect-gpt |
| **Secondary content** | DPO has massive secondary content (dozens of blog posts, HuggingFace tutorials, Together.ai explainer); DetectGPT covered by Stanford Engineering news |
| **Current role** | OpenAI (Post-training Frontiers team co-lead) |
| **Name uniqueness** | **VERY COMMON NAME** -- "Eric Mitchell" returns a film director (Wikipedia), accident news, IMDB pages, criminal records, etc. The ML researcher is buried among many Eric Mitchells. |
| **Key recognition driver (or lack thereof)** | DPO is one of the most impactful ML papers of 2023, but Eric Mitchell is listed 3rd author (Rafael Rafailov is 1st). The common name means web searches return many non-ML results. Despite 15K citations, the name ambiguity likely confuses LLMs. |

### Karan Goel (3,142 citations, 57% recognition)

| Dimension | Finding |
|-----------|---------|
| **Personal site** | Yes -- krandiash.github.io |
| **Wikipedia** | No |
| **Google Scholar** | ~3,142 citations |
| **Industry** | Founder & CEO of Cartesia ($100M raised, Kleiner Perkins, NVIDIA backing) |
| **GitHub** | Active profile |
| **Secondary content** | TED AI speaker; India.com feature; Amplify Partners podcast; Dell Technologies Capital profile; Crunchbase |
| **Key papers** | Co-author of S4 (structured state space models) with Albert Gu |
| **Name uniqueness** | "Karan Goel" is a moderately common Indian name, but the startup CEO visibility helps |
| **Key recognition driver (or lack thereof)** | Despite being CEO of a well-funded startup and co-author of S4, he is somewhat overshadowed by Albert Gu (who gets TIME 100 and Wikipedia mentions for the same work). The S4/Mamba narrative centers on Gu, not Goel. |

### Sang Michael Xie (5,305 citations, 57% recognition)

| Dimension | Finding |
|-----------|---------|
| **Personal site** | Yes -- cs.stanford.edu/~eix/ |
| **Wikipedia** | No |
| **Google Scholar** | ~5,305 citations |
| **GitHub** | sangmichaelxie -- repos for DoReMi, pretraining analysis |
| **Current role** | OpenAI (formerly Meta GenAI, LLaMA team) |
| **Secondary content** | Limited compared to peers -- mostly academic papers and profiles |
| **Key papers** | Poverty mapping with satellite imagery; DoReMi (data mixing for pretraining) |
| **Name uniqueness** | Unusual compound name "Sang Michael Xie" -- fairly unique when full name is used, but "Xie" alone is very common |
| **Key recognition driver (or lack thereof)** | His work is impactful but mostly "infrastructure" contributions to larger projects (LLaMA team, data mixing). He lacks a single viral, name-associated project like FlashAttention or DPO. No startup, no TIME list, no named software tool. |

**Cell 6 Summary:** The 57% recognition group shares a pattern: they have solid academic work but lack a single "viral" association. Eric Mitchell's DPO is viral but he's a middle author with an extremely common name. Karan Goel co-created S4 but Albert Gu gets the credit. Sang Michael Xie contributes to major projects but none bear his name. **Name uniqueness and first-author association with a named tool/method appear critical.**

---

## Cell 7: ML + Low Citations + Recognized (43-57%)

### Mayee Chen (215 citations, 57% recognition)

| Dimension | Finding |
|-----------|---------|
| **Personal site** | Yes -- mayeechen.github.io |
| **Wikipedia** | No |
| **Google Scholar** | ~1,356 citations (higher than OpenAlex) |
| **Lab affiliation** | Stanford Hazy Research (Chris Re's lab) and Stanford DAWN |
| **Secondary content** | Lecture notes on weak supervision (widely referenced); Snorkel AI blog post; Stanford SAIL affiliation |
| **Twitter/X** | Active academic Twitter presence |
| **Current role** | Research intern at Allen Institute for AI (AI2) working on OLMo |
| **Key papers** | Embroid (NeurIPS 2023), Smoothie (NeurIPS 2024), weak supervision methods |
| **Name uniqueness** | "Mayee Chen" is fairly unique -- "Mayee" is distinctive |
| **Key recognition driver** | Hazy Research lab prestige (Chris Re's lab); Stanford AI ecosystem generates many cross-references; weak supervision is a well-known topic; AI2/OLMo connection |

### Sabri Eyuboglu (267 citations, 57% recognition)

| Dimension | Finding |
|-----------|---------|
| **Personal site** | Yes -- sabrieyuboglu.com |
| **Wikipedia** | No |
| **Google Scholar** | ~291 citations |
| **Lab affiliation** | Stanford Hazy Research (Chris Re's lab) + Stanford HAI |
| **GitHub** | seyuboglu -- HazyResearch/domino published as open-source |
| **Secondary content** | Domino blog post on Hazy Research blog + Stanford SAIL blog; Cartesia connection |
| **Industry** | Connected to Cartesia (contributor/researcher) |
| **Key papers** | Domino (ICLR 2022) -- systematic error discovery |
| **Name uniqueness** | Very unique -- "Sabri Eyuboglu" returns only this person |
| **Key recognition driver** | Unique name + Hazy Research lab + Domino project with blog posts. The Cartesia ecosystem (Gu, Goel, Eyuboglu) creates mutual cross-references. |

### Lisa Dunlap (303 citations, 43% recognition)

| Dimension | Finding |
|-----------|---------|
| **Personal site** | Yes -- lisadunlap.com |
| **Wikipedia** | No |
| **Google Scholar** | ~1,575 citations (much higher than OpenAlex) |
| **Lab affiliation** | UC Berkeley RISE Lab + BerkeleyVL (advisors: Joseph Gonzalez, Trevor Darrell, Jacob Steinhardt) |
| **GitHub** | lisadunlap -- neural-backed-decision-trees; VPBench |
| **Secondary content** | VPBench blog post (lisadunlap.github.io/vpbench/); Weights & Biases portfolio |
| **Twitter/X** | Active academic presence |
| **Key papers** | VPBench (VLM benchmark fragility); dissertation "Understanding Models When Everything Is a Vibe" |
| **Name uniqueness** | "Lisa Dunlap" is a moderately common American name |
| **Key recognition driver** | Berkeley AI ecosystem (Gonzalez, Darrell, Steinhardt are all famous); VLM evaluation is a hot topic; but name is somewhat common and she's still a PhD student |

**Cell 7 Summary:** These researchers have low citation counts but are recognized at 43-57% because they are embedded in elite lab ecosystems (Stanford Hazy Research, Berkeley RISE/VL) that generate enormous cross-referencing content. Lab blog posts, joint papers with famous advisors, and participation in high-profile projects (OLMo, Cartesia, VLM evaluation) create web presence disproportionate to their individual citation counts.

---

## Cell 8: ML + Low Citations + NOT Recognized

No such researchers exist in our sample. This is itself a significant finding: **in the ML/AI subfield, even junior researchers with <300 citations achieve 43%+ recognition.** This is likely because:
1. ML/AI is the most heavily represented topic in LLM training data
2. Elite ML labs (Stanford, Berkeley, CMU) have well-indexed blog posts, news, and social media
3. The ML community has extremely active Twitter/X, blog, and GitHub ecosystems
4. Papers in ML get explained in blog posts, YouTube videos, and tutorials at much higher rates than papers in other CS subfields

---

## Cross-Cutting Findings

### 1. Name Collision is a Major Confound

Three of the "unrecognized high-citation" researchers (Yan Jiao, Xinming Wang, Sylvie Dujardin) turned out to be OpenAlex disambiguation errors. The citations attributed to them actually belong to researchers in chemistry, atmospheric science, and neuroscience. **Any analysis using OpenAlex citation counts for common Chinese names must be verified.**

| Researcher | OpenAlex Citations | Actual CS Citations | OpenAlex Field | True Field |
|------------|-------------------|--------------------|----|---|
| Yan Jiao | 46,331 | <2,000 (est.) | "computer networking" | Chemistry (Adelaide) |
| Xinming Wang | 44,522 | <500 (est.) | "HCI" | Atmospheric chemistry (CAS) |
| Sylvie Dujardin | 149 | 0 (not a CS researcher) | "operating systems" | Sleep medicine |
| Eric Mitchell | 74,659 (OpenAlex) | ~15,338 (GScholar) | ML (correct) | ML (correct but merged profiles) |

### 2. The Recognition Formula

Based on this analysis, LLM recognition appears driven by a hierarchy of factors:

| Factor | Impact | Examples |
|--------|--------|----------|
| **Wikipedia page** | Very high | Mario Gerla (89%), Mamba architecture page helps Albert Gu (86%) |
| **Named, widely-used software** | Very high | FlashAttention/Tri Dao (100%), SVMlight/Joachims (97%) |
| **Named method in wide use** | High | DPO helps Eric Mitchell (57%) despite common name |
| **Startup/industry visibility** | High | Cartesia helps Albert Gu (86%), Karan Goel (57%) |
| **YouTube/educational content** | Medium-high | Micinski (59%), Lysecky/zyBooks (54%) |
| **Elite lab blog posts** | Medium | Hazy Research blog helps Mayee Chen (57%), Eyuboglu (57%) |
| **Web3/crypto ecosystem** | Medium | IPFS/Protocol Labs helps Psaras (69%) |
| **Prestigious fellowships** | Medium | Multiple fellowships help Raghunathan (100%) |
| **Name uniqueness** | Multiplicative | Unique names (Eyuboglu, Raghunathan, Psaras) amplify all signals; common names (Eric Mitchell, Xinming Wang) attenuate them |

### 3. Citation Count vs. Web Footprint

| Researcher | Citations | Recognition | Primary Web Signal |
|------------|-----------|-------------|-------------------|
| Tri Dao | 2,817 | 100% | 23K GitHub stars, universal LLM adoption |
| Aditi Raghunathan | 4,689 | 100% | Unique name + many fellowship pages |
| Thorsten Joachims | 52,645 | 97% | SVMlight software + VP for AI at Cornell |
| Mario Gerla | 54,351 | 89% | Wikipedia + ARPANET history |
| Albert Gu | 3,015 | 86% | TIME 100 AI + Wikipedia (Mamba) + Cartesia |
| Yiannis Psaras | 318 | 69% | IPFS/Protocol Labs ecosystem |
| Kristopher Micinski | 850 | 59% | YouTube lectures + blog |
| Karan Goel | 3,142 | 57% | Cartesia CEO but overshadowed by Gu |
| Eric Mitchell | ~15,338 | 57% | DPO but common name + middle author |
| Sang Michael Xie | 5,305 | 57% | Solid work but no named tool/project |
| Mayee Chen | 215 | 57% | Hazy Research ecosystem |
| Sabri Eyuboglu | 267 | 57% | Unique name + Hazy Research + Domino |
| Susan Lysecky | 382 | 54% | zyBooks in thousands of course syllabi |
| Lisa Dunlap | 303 | 43% | Berkeley ecosystem but common-ish name |
| Qiansheng Rao | 15 | 0% | Virtually no web presence |

### 4. The ML vs. Systems Gap

The minimum recognition rate for ML researchers in our sample is 43% (Lisa Dunlap, 303 citations). For systems researchers, the minimum for recognized researchers is 54% (Susan Lysecky, 382 citations), but there exist systems researchers with 0% recognition even at moderate citation levels. This confirms that the ML/AI field has a "recognition floor" -- the field itself is so heavily represented in training data that even junior ML researchers inherit visibility through lab affiliations and community discourse.

### 5. What Separates 57% from 100%?

The jump from 57% to 86-100% recognition requires at least one of:
- A Wikipedia page (Gerla, Albert Gu via Mamba)
- A named software tool with >10K GitHub stars (Tri Dao)
- A perfectly unique name combined with multiple prestigious awards (Raghunathan)
- A named method that became an industry standard (Joachims/SVMlight, Tri Dao/FlashAttention)

Researchers at 57% typically have solid academic presence but lack a single "anchor" that makes them unambiguously findable in web text.
