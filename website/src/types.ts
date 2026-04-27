export type Tier = "T1" | "T2" | "T3" | "T4" | "T5" | "T6" | "T7";

export interface ModelSummary {
  model: string;
  vendor: string | null;
  family: string | null;
  type: "open" | "proprietary" | string | null;
  arch: "dense" | "moe" | string | null;
  params_B: number | null;
  active_B: number | null;
  thinking: boolean;
  accuracy: number;
  raw_accuracy: number;
  tier_accuracy: Record<Tier, number>;
  tier_stats: Record<Tier, { correct: number; wrong: number; refusal: number; total: number }>;
}

export interface ProbeMeta {
  id: string;
  tier: Tier;
  domain: string | null;
  source_type: string | null;
  question: string;
  answer: string;
  n_models?: number;
  correct_rate?: number;
  halluc_rate?: number;
  refusal_rate?: number;
}

export interface ProbeResponse {
  response: string;
  verdict: "CORRECT" | "WRONG" | "REFUSAL" | string;
  correct: boolean;
  refusal: boolean;
}

export interface ResearcherEvidence {
  primary_subfield: string | null;
  secondary_subfields: string[];
  affiliations: string[];
  venues: string[];
  named_systems: string[];
  co_authors: string[];
  top_works: { title: string; year: number; cited: number }[];
}

export interface TierFile {
  tier: Tier;
  probes: Record<string, {
    question: string;
    answer: string;
    domain: string | null;
    source_type: string | null;
    responses: Record<string, ProbeResponse>;
    evidence?: ResearcherEvidence;
  }>;
}

export interface CalibrationPoint {
  model: string;
  params_B: number;
  active_B: number | null;
  accuracy: number;
  raw_accuracy: number;
  vendor: string;
  family: string;
  arch: string;
  thinking: boolean;
}

export interface ProprietaryEstimate {
  model: string;
  accuracy: number;
  raw_accuracy: number;
  vendor: string;
  family: string;
  thinking: boolean;
  // Two-regime scheme:
  //   regime = "pretraining":  estimated_B is the single-regime inversion;
  //                            estimated_B_eff === estimated_B
  //   regime = "distilled":    estimated_B is the actual-parameter estimate
  //                            (= estimated_B_eff / DISTILL_BOOST);
  //                            estimated_B_eff is the uncorrected effective
  //                            capacity (raw single-regime inversion).
  estimated_B: number;
  estimated_B_eff: number;
  regime: "pretraining" | "distilled";
  distill_anchor?: string;
  pi_lo: number;
  pi_hi: number;
}

export interface LooPrediction {
  model: string;
  vendor: string | null;
  arch: string | null;
  actual_B: number;
  pred_B: number;
  actual_acc: number;
  pred_acc: number;
  fold_err: number | null;
}

export interface MoeFit { slope: number; intercept: number; r_squared: number }
export interface MoePoint {
  model: string;
  params_B: number;
  active_B: number;
  accuracy: number;
  vendor: string | null;
  thinking: boolean;
}

export interface CalibrationData {
  fit: {
    slope: number;
    intercept: number;
    r_squared: number;
    residual_se: number;
    pi_half_log10?: number;
    pi_factor?: number;
  };
  loo_cv: {
    r_squared: number;
    median_fold_err: number;
    within_2x: number;
    within_3x: number;
    predictions: LooPrediction[];
  };
  n_calibration: number;
  n_proprietary: number;
  vendors: string[];
  calibration_points: CalibrationPoint[];
  excluded_points: { model: string; params_B: number; accuracy: number; vendor: string; reason: string }[];
  proprietary_estimates: ProprietaryEstimate[];
  distillation?: {
    boost: number;
    boost_range: [number, number];
    anchor_pair: [string, string];
    students: string[];
  };
  moe: {
    total: MoeFit | null;
    active: MoeFit | null;
    points: MoePoint[];
  };
}

export interface FingerprintPair {
  pair_id?: string;
  a: string;
  b: string;
  jaccard: number;
  lift?: number;
  hss: number;
  n_a?: number;
  n_b?: number;
  inter?: number;
  both_wrong: number;
  same_wrong: number;
  class: string;
  family?: string;
  vendor_a?: string;
  vendor_b?: string;
}

export interface FingerprintData {
  n_probes: number;
  n_models: number;
  thresholds: { shared_base_hss: number; lineage_hss: number; min_both_wrong: number };
  families: { family: string; pairs: FingerprintPair[] }[];
  cross_vendor: FingerprintPair[];
  heatmap: {
    models: string[];
    matrix: ({ j: number | null; hss: number | null; both_w: number | null } | null)[][];
  };
  consecutive_pairs: FingerprintPair[];
}

export interface DensingPoint {
  model: string;
  vendor: string;
  family: string;
  arch: string;
  thinking: boolean;
  params_B: number;
  active_B: number;
  release_date: string;
  pen_acc: number;
  raw_acc: number;
  log10_params: number;
  months: number;
}

export interface DensingData {
  n: number;
  fit: {
    b0: number;
    b1_logN: number;
    b2_time: number;
    r_squared: number;
    ci95_b2: [number, number];
    baseline_b1: number;
    baseline_r_squared: number;
    r2_gain_from_time: number;
  };
  densing_prediction: { b2: number; doubling_time_months: number; comment: string };
  points: DensingPoint[];
  partial_residuals: {
    model: string;
    months: number;
    resid: number;
    log10_params: number;
    vendor: string;
    thinking: boolean;
  }[];
}

export interface BenchmarkFit {
  n?: number;
  slope: number;
  intercept: number;
  r2: number;
}

export interface BenchmarkJoint {
  intercept: number;
  slope_params: number;
  slope_months: number;
  r2: number;
}

export interface BenchmarkPoint {
  model: string;
  vendor: string;
  log10_params: number;
  months: number;
  release_date: string;
  score: number;
  ikp: number;
}

export interface BenchmarkEntry {
  key: string;
  label: string;
  n: number;
  benchmark_fit: BenchmarkFit;
  ikp_fit_same_subset: BenchmarkFit;
  benchmark_joint: BenchmarkJoint;
  ikp_joint_same_subset: BenchmarkJoint;
  points: BenchmarkPoint[];
}

export interface BenchmarksData {
  n_total: number;
  ikp_full_fit: BenchmarkFit;
  ikp_full_joint: BenchmarkJoint;
  benchmarks: BenchmarkEntry[];
}

export interface RecognitionPoint {
  probe_id: string;
  name: string;
  tier: Tier;
  recognition_rate: number;
  correct: number;
  total: number;
  refusal: number;
  wrong: number;
  field: string | null;
  domain: string | null;
  works_count: number | null;
  cited_by_count: number | null;
  h_index: number | null;
  i10_index: number | null;
}

export interface RecognitionData {
  n: number;
  n_with_citations: number;
  pearson_log_citations: number;
  spearman_log_citations: number;
  quintile_buckets: {
    index: number;
    n: number;
    log_cit_range: [number, number];
    citations_range: [number, number];
    median_citations: number;
    median_recognition: number;
  }[];
  points: RecognitionPoint[];
}

export interface HallucinationData {
  per_model: {
    model: string;
    vendor: string | null;
    thinking: boolean;
    accuracy: number;
    t5_t7_wrong: number;
    t5_t7_total: number;
    halluc_rate: number;
    per_tier: Record<"T5" | "T6" | "T7", number | null>;
  }[];
  vendors: {
    vendor: string;
    n_models: number;
    mean_halluc: number;
    median_halluc: number;
    min: number;
    max: number;
  }[];
}

export interface GenerationFamily {
  family: string;
  chain: {
    model: string;
    vendor: string | null;
    accuracy: number;
    raw_accuracy: number;
    tier_accuracy: Record<Tier, number>;
    params_B: number | null;
  }[];
}

export interface GenerationsData {
  families: GenerationFamily[];
  gpt5_family: {
    model: string;
    variant: "base" | "mini" | "nano" | "pro" | "think" | string;
    accuracy: number;
    raw_accuracy: number;
    tier_accuracy: Record<Tier, number>;
  }[];
}

export interface ModelDetail extends ModelSummary {
  samples: Record<Tier, {
    CORRECT: { probe_id: string; question: string; gold_answer: string; model_response: string }[];
    WRONG: { probe_id: string; question: string; gold_answer: string; model_response: string }[];
    REFUSAL: { probe_id: string; question: string; gold_answer: string; model_response: string }[];
  }>;
}

export interface ThinkingPair {
  base: string;
  think: string;
  base_acc: number;
  think_acc: number;
  delta: number;
  vendor: string;
  params_B: number | null;
}

export interface PipelineStage {
  id: string;
  name: string;
  description: string;
  output: string;
  count: string;
  script: string;
}

// ── Fingerprint pair-detail page ────────────────────────────────────────────
export interface PairJointWrong {
  probe_id: string;
  tier: Tier;
  domain?: string;
  question: string;
  gold_answer: string;
  response_a: string | null;
  response_b: string | null;
  verdict_a: string;
  verdict_b: string;
  same_wrong: boolean;
}

export interface PairDisagreement {
  probe_id: string;
  tier: Tier;
  question: string;
  gold_answer: string;
  correct_a: boolean;
  correct_b: boolean;
  response_a: string | null;
  response_b: string | null;
  verdict_a: string;
  verdict_b: string;
}

export interface FingerprintPairDetail {
  pair_id: string;
  a: string;
  b: string;
  family?: string;
  vendor_a?: string;
  vendor_b?: string;
  hss: number;
  jaccard: number;
  lift?: number;
  both_wrong: number;
  same_wrong: number;
  class: string;
  sources: ("consecutive" | "cross_vendor" | string)[];
  n_joint_wrong: number;
  n_same_wrong: number;
  joint_wrong: PairJointWrong[];
  disagreement: PairDisagreement[];
}
