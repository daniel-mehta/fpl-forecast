export function formatNumber(value: string | number | undefined, digits = 2): string {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric.toFixed(digits) : "Not available";
}

export function formatPrice(value: string | number | undefined): string {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? `£${(numeric / 10).toFixed(1)}m` : "Not available";
}

export function formatPercentage(value: string | number | undefined): string {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? `${Math.round(numeric * 100)}%` : "Not available";
}

export function formatTeam(value: string | undefined): string {
  if (!value) {
    return "Not available";
  }
  return value
    .replace(/^team_/, "")
    .split("_")
    .map((part) => `${part.charAt(0).toUpperCase()}${part.slice(1)}`)
    .join(" ");
}

export function formatLabel(value: string | undefined): string {
  if (!value) {
    return "Not available";
  }
  return value
    .toLowerCase()
    .split("_")
    .map((part) => `${part.charAt(0).toUpperCase()}${part.slice(1)}`)
    .join(" ");
}

export function formatDate(value: unknown): string {
  if (typeof value !== "string") {
    return "Not available";
  }
  const date = new Date(value);
  return Number.isNaN(date.valueOf())
    ? "Not available"
    : new Intl.DateTimeFormat(undefined, {
        dateStyle: "medium",
        timeStyle: "short",
      }).format(date);
}

export function formatPlayerStatus(value: string | undefined): string {
  const labels: Record<string, string> = {
    a: "Available",
    d: "Doubtful",
    i: "Injured",
    n: "Unavailable",
    s: "Suspended",
    u: "Unavailable",
  };
  return value ? (labels[value.toLowerCase()] ?? formatLabel(value)) : "Not available";
}

export function formatComparisonValue(column: string, value: string): string {
  if (column.includes("model")) {
    return formatLabel(value);
  }
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return formatLabel(value);
  }
  return Number.isInteger(numeric) ? String(numeric) : numeric.toFixed(3);
}
