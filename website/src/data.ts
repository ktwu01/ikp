import { useEffect, useState } from "react";
import type {
  BenchmarksData,
  CalibrationData,
  DensingData,
  FingerprintData,
  GenerationsData,
  HallucinationData,
  ModelDetail,
  ModelSummary,
  PipelineStage,
  ProbeMeta,
  RecognitionData,
  SensitivityData,
  ThinkingPair,
  Tier,
  TierFile,
} from "./types";

const DATA = "data";

// Simple in-memory cache so navigating back doesn't re-fetch.
const cache = new Map<string, unknown>();

async function fetchJson<T>(path: string): Promise<T> {
  if (cache.has(path)) return cache.get(path) as T;
  const res = await fetch(path);
  if (!res.ok) throw new Error(`Failed ${res.status}: ${path}`);
  const data = (await res.json()) as T;
  cache.set(path, data);
  return data;
}

export function useFetch<T>(path: string): { data: T | null; error: Error | null; loading: boolean } {
  const [data, setData] = useState<T | null>(() => (cache.get(path) as T | undefined) ?? null);
  const [error, setError] = useState<Error | null>(null);
  const [loading, setLoading] = useState(!cache.has(path));

  useEffect(() => {
    if (cache.has(path)) {
      setData(cache.get(path) as T);
      setLoading(false);
      return;
    }
    let alive = true;
    setLoading(true);
    fetchJson<T>(path)
      .then((d) => alive && (setData(d), setLoading(false)))
      .catch((e) => alive && (setError(e), setLoading(false)));
    return () => {
      alive = false;
    };
  }, [path]);

  return { data, error, loading };
}

export const useModels = () => useFetch<ModelSummary[]>(`${DATA}/models.json`);
export const useProbes = () => useFetch<ProbeMeta[]>(`${DATA}/probes.json`);
export const useCalibration = () => useFetch<CalibrationData>(`${DATA}/calibration.json`);
export const useSensitivity = () => useFetch<SensitivityData>(`${DATA}/sensitivity.json`);
export const usePipeline = () => useFetch<{ stages: PipelineStage[] }>(`${DATA}/pipeline.json`);
export const useThinkingPairs = () => useFetch<ThinkingPair[]>(`${DATA}/thinking_pairs.json`);
export const useModelDetail = (model: string) => useFetch<ModelDetail>(`${DATA}/models/${model}.json`);
export const useTier = (tier: Tier) => useFetch<TierFile>(`${DATA}/tiers/${tier}.json`);
export const useFingerprint = () => useFetch<FingerprintData>(`${DATA}/fingerprint.json`);
export const useDensing = () => useFetch<DensingData>(`${DATA}/densing.json`);
export const useBenchmarks = () => useFetch<BenchmarksData>(`${DATA}/benchmarks.json`);
export const useRecognition = () => useFetch<RecognitionData>(`${DATA}/recognition.json`);
export const useHallucination = () => useFetch<HallucinationData>(`${DATA}/hallucination.json`);
export const useGenerations = () => useFetch<GenerationsData>(`${DATA}/generations.json`);
