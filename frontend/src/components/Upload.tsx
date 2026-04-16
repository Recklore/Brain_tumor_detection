import { useCallback, useMemo, useState } from "react";
import { toast } from "react-toastify";
import { BACKEND_URL } from "../config/config";
import {
  Activity,
  AlertTriangle,
  Brain,
  CheckCircle2,
  Loader2,
  RotateCcw,
  UploadCloud,
} from "lucide-react";
import Button from "@mui/joy/Button";
import axios from "axios";

type ClassProbabilities = Record<string, number>;

interface Detection {
  box: [number, number, number, number];
  label: number;
  class_name: string;
  score: number;
  cam_peak: number;
}

interface PredictResponse {
  prediction: string;
  confidence: number;
  class_probabilities: ClassProbabilities;
  active_classes: string[];
  detector_label_map: Record<string, number>;
  detections: Detection[];
  annotated_image_mime: string;
  annotated_image_base64: string;
  report?: StructuredReport | null;
  report_status?: string;
}

interface StructuredReport {
  diagnosis_summary: string;
  risk_level: "low" | "moderate" | "high";
  key_findings: string[];
  recommendations: string[];
  confidence_note: string;
  disclaimer: string;
}

function toTitleCase(value: string): string {
  return value
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(2)}%`;
}

function toTitleWord(value: string): string {
  return value.charAt(0).toUpperCase() + value.slice(1);
}

function reportUnavailableMessage(status?: string): string {
  if (!status) return "Structured report is currently unavailable.";
  switch (status) {
    case "unavailable_ollama_not_configured":
      return "Structured report unavailable: Ollama is not configured.";
    case "unavailable_ollama_connection_failed":
      return "Structured report unavailable: unable to connect to Ollama.";
    case "unavailable_ollama_model_not_found":
      return "Structured report unavailable: Ollama model not found.";
    case "unavailable_ollama_invalid_output":
      return "Structured report unavailable: Ollama returned invalid output.";
    case "unavailable_ollama_generation_failed":
      return "Structured report unavailable: Ollama generation failed. Please retry.";
    case "unavailable_generation_failed":
      return "Structured report unavailable due to generation failure. Please retry.";
    default:
      return `Structured report unavailable (${status}).`;
  }
}

function classColor(className: string): string {
  const key = className.toLowerCase();
  if (key.includes("glioma")) return "bg-red-500";
  if (key.includes("meningioma")) return "bg-amber-500";
  if (key.includes("pituitary")) return "bg-blue-500";
  if (key.includes("notumor")) return "bg-emerald-500";
  return "bg-rose-500";
}

export default function Upload() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictResponse | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files?.[0] ?? null;
    setFile(selected);
    setResult(null);

    if (!selected) {
      setPreview(null);
      return;
    }

    if (selected.type.startsWith("image/")) {
      setPreview(URL.createObjectURL(selected));
      return;
    }

    setPreview(null);
    toast.error("Please upload an image file (JPG, JPEG, PNG).");
  };

  const handleSubmit = useCallback(async () => {
    if (!file) {
      toast.error("Please select an MRI scan first.");
      return;
    }

    try {
      setLoading(true);
      const formData = new FormData();
      formData.append("file", file);

      const response = await axios.post<PredictResponse>(
        `${BACKEND_URL}/api/predict`,
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
          timeout: 120000,
        }
      );

      setResult(response.data);
      toast.success("Analysis complete.");
    } catch (error) {
      if (axios.isAxiosError(error)) {
        const apiMessage =
          (error.response?.data as { error?: string } | undefined)?.error ??
          "Prediction failed. Please try again.";
        toast.error(apiMessage);
      } else {
        toast.error("Prediction failed. Please try again.");
      }
    } finally {
      setLoading(false);
    }
  }, [file]);

  const handleReset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
  };

  if (result) {
    return <ResultView result={result} rawPreview={preview} onReset={handleReset} />;
  }

  return (
    <div className="w-full max-w-5xl mx-auto py-10 px-4">
      <div className="flex flex-col gap-2 mb-8">
        <h1 className="text-3xl sm:text-4xl font-bold from-red-600 to-rose-500 leading-tight bg-gradient-to-r bg-clip-text text-transparent">
          Brain Tumor MRI Analysis
        </h1>
        <p className="text-zinc-600 text-lg max-w-3xl font-medium">
          Upload an MRI scan to classify tumor type and return a Grad-CAM annotated image with
          tumor bounding boxes.
        </p>
      </div>

      <div className="bg-white rounded-2xl shadow-sm border border-zinc-200 p-6 sm:p-8">
        <label className="text-base font-semibold text-zinc-800">Upload MRI Scan</label>
        <p className="text-sm text-zinc-500 mb-4">Supported formats: JPG, JPEG, PNG</p>

        <div className="relative">
          <div
            className={`border-2 border-dashed rounded-2xl p-10 text-center transition-colors cursor-pointer group ${
              file
                ? "border-red-400 bg-red-50/50"
                : "border-zinc-300 hover:border-red-400 hover:bg-red-50/30"
            }`}
          >
            <input
              type="file"
              accept=".jpg,.jpeg,.png"
              onChange={handleFileChange}
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
            />
            <div className="flex flex-col items-center gap-4 relative">
              {preview ? (
                <div className="relative w-40 h-40 rounded-xl overflow-hidden border-2 border-red-200 shadow-md">
                  <img src={preview} alt="MRI preview" className="w-full h-full object-cover" />
                  {loading && (
                    <>
                      <div className="absolute inset-0 bg-red-500/10" />
                      <div className="scan-line absolute left-0 right-0 h-1 bg-gradient-to-r from-transparent via-red-400 to-transparent shadow-[0_0_12px_rgba(239,68,68,0.9)]" />
                    </>
                  )}
                </div>
              ) : (
                <div className="p-5 bg-red-100 rounded-full group-hover:scale-110 transition-transform">
                  <UploadCloud className="h-9 w-9 text-red-600" />
                </div>
              )}

              <div className="space-y-1">
                <p className="text-lg font-medium text-zinc-800">
                  {file ? file.name : "Click to upload or drag and drop"}
                </p>
                <p className="text-sm text-zinc-500">
                  {file ? `${(file.size / 1024).toFixed(1)} KB` : "JPG, JPEG, PNG"}
                </p>
              </div>
            </div>
          </div>
        </div>

        <div className="flex flex-col sm:flex-row justify-end gap-3 pt-6">
          {file && !loading && (
            <Button variant="outlined" color="neutral" onClick={handleReset} size="lg">
              Clear
            </Button>
          )}
          <Button
            onClick={handleSubmit}
            size="lg"
            disabled={loading || !file}
            sx={{
              background: "linear-gradient(135deg, #dc2626 0%, #e11d48 100%)",
              "&:hover": { background: "linear-gradient(135deg, #b91c1c 0%, #be123c 100%)" },
            }}
          >
            {loading ? (
              <>
                <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                Analyzing...
              </>
            ) : (
              <>
                <Activity className="mr-2 h-5 w-5" />
                Analyze Scan
              </>
            )}
          </Button>
        </div>

        {loading && (
          <div className="mt-6 text-center text-sm text-red-700 fade-in-up">
            Running inference and generating Grad-CAM tumor boxes...
          </div>
        )}
      </div>
    </div>
  );
}

function ResultView({
  result,
  rawPreview,
  onReset,
}: {
  result: PredictResponse;
  rawPreview: string | null;
  onReset: () => void;
}) {
  const annotatedSrc = `data:${result.annotated_image_mime};base64,${result.annotated_image_base64}`;
  const report = result.report ?? null;

  const sortedProbabilities = useMemo(
    () =>
      Object.entries(result.class_probabilities).sort((a, b) => b[1] - a[1]),
    [result.class_probabilities]
  );

  return (
    <div className="w-full max-w-6xl mx-auto py-8 px-4 fade-in-up">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-6">
        <div className="flex items-start gap-3">
          <div className="p-2.5 rounded-lg bg-emerald-100 text-emerald-700">
            <CheckCircle2 className="h-5 w-5" />
          </div>
          <div>
            <h1 className="text-2xl sm:text-3xl font-bold text-zinc-900">Analysis Complete</h1>
            <p className="text-sm text-zinc-500 mt-1">
              Prediction: <span className="font-semibold">{toTitleCase(result.prediction)}</span> · Confidence: {formatPercent(result.confidence)}
            </p>
          </div>
        </div>

        <Button
          onClick={onReset}
          size="md"
          sx={{
            background: "linear-gradient(135deg, #dc2626 0%, #e11d48 100%)",
            "&:hover": { background: "linear-gradient(135deg, #b91c1c 0%, #be123c 100%)" },
          }}
        >
          <RotateCcw className="h-4 w-4 mr-2" /> New Scan
        </Button>
      </div>

      <div className="grid lg:grid-cols-5 gap-6">
        <div className="lg:col-span-3 bg-white rounded-2xl shadow-sm border border-zinc-200 overflow-hidden">
          <div className="px-5 py-3 border-b border-zinc-100 flex items-center justify-between">
            <h3 className="font-semibold text-zinc-800 flex items-center gap-2">
              <Brain className="h-4 w-4 text-red-600" /> Annotated MRI (Grad-CAM Boxes)
            </h3>
            <span className="text-xs text-zinc-500">{result.detections.length} detection(s)</span>
          </div>
          <div className="bg-black aspect-square sm:aspect-[4/3] flex items-center justify-center">
            <img
              src={annotatedSrc || rawPreview || ""}
              alt="Annotated MRI"
              className="max-h-full max-w-full object-contain"
            />
          </div>
        </div>

        <div className="lg:col-span-2 space-y-4">
          <div className="bg-white rounded-xl shadow-sm border border-zinc-200 p-4">
            <h3 className="font-semibold text-zinc-800 mb-3">Class Probabilities</h3>
            <div className="space-y-3">
              {sortedProbabilities.map(([className, score]) => (
                <div key={className}>
                  <div className="flex items-center justify-between text-sm mb-1.5">
                    <span className="flex items-center gap-2 text-zinc-700">
                      <span className={`w-2.5 h-2.5 rounded-full ${classColor(className)}`} />
                      <span>{toTitleCase(className)}</span>
                    </span>
                    <span className="font-mono text-zinc-700">{formatPercent(score)}</span>
                  </div>
                  <div className="h-2 bg-zinc-100 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full ${classColor(className)} transition-all duration-700`}
                      style={{ width: `${Math.max(0, Math.min(100, score * 100))}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-white rounded-xl shadow-sm border border-zinc-200 p-4">
            <h3 className="font-semibold text-zinc-800 mb-3">Tumor Report</h3>
            {report ? (
              <div className="space-y-3 text-sm text-zinc-700">
                <div>
                  <span className="font-semibold text-zinc-900">Diagnosis:</span> {report.diagnosis_summary}
                </div>
                <div>
                  <span className="font-semibold text-zinc-900">Risk:</span> {toTitleWord(report.risk_level)}
                </div>
                <div>
                  <span className="font-semibold text-zinc-900">Confidence Note:</span> {report.confidence_note}
                </div>
                {report.key_findings.length > 0 && (
                  <div>
                    <span className="font-semibold text-zinc-900">Key Findings:</span>
                    <ul className="mt-1 space-y-1 text-zinc-600">
                      {report.key_findings.map((finding, index) => (
                        <li key={`${finding}-${index}`}>- {finding}</li>
                      ))}
                    </ul>
                  </div>
                )}
                {report.recommendations.length > 0 && (
                  <div>
                    <span className="font-semibold text-zinc-900">Recommendations:</span>
                    <ul className="mt-1 space-y-1 text-zinc-600">
                      {report.recommendations.map((recommendation, index) => (
                        <li key={`${recommendation}-${index}`}>- {recommendation}</li>
                      ))}
                    </ul>
                  </div>
                )}
                <div className="text-xs text-zinc-500 border-t border-zinc-100 pt-2">{report.disclaimer}</div>
              </div>
            ) : (
              <div className="flex items-start gap-2 text-sm text-zinc-600">
                <AlertTriangle className="h-4 w-4 mt-0.5 text-amber-500" />
                <span>{reportUnavailableMessage(result.report_status)}</span>
              </div>
            )}
          </div>

          <div className="bg-white rounded-xl shadow-sm border border-zinc-200 p-4">
            <h3 className="font-semibold text-zinc-800 mb-3">Detected Tumor Boxes</h3>
            {result.detections.length === 0 ? (
              <div className="flex items-start gap-2 text-sm text-zinc-600">
                <AlertTriangle className="h-4 w-4 mt-0.5 text-amber-500" />
                <span>No tumor boxes were detected for this scan.</span>
              </div>
            ) : (
              <div className="space-y-2">
                {result.detections.map((det, index) => {
                  const width = det.box[2] - det.box[0];
                  const height = det.box[3] - det.box[1];
                  return (
                    <div
                      key={`${det.class_name}-${index}-${det.box.join("-")}`}
                      className="rounded-lg border border-zinc-200 p-3"
                    >
                      <div className="flex items-center justify-between text-sm">
                        <span className="font-semibold text-zinc-800">
                          {toTitleCase(det.class_name)}
                        </span>
                        <span className="text-zinc-500">Score: {formatPercent(det.score)}</span>
                      </div>
                      <div className="text-xs text-zinc-500 mt-1">
                        Box: [{det.box[0]}, {det.box[1]}, {det.box[2]}, {det.box[3]}] · Size: {width}x{height}px · CAM peak: {det.cam_peak.toFixed(3)}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
