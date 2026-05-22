{{/*
Expand the name of the chart.
*/}}
{{- define "pr-review-ai.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Fullname: release-name + chart-name, truncated to 63 chars.
*/}}
{{- define "pr-review-ai.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Common labels.
*/}}
{{- define "pr-review-ai.labels" -}}
helm.sh/chart: {{ include "pr-review-ai.name" . }}-{{ .Chart.Version | replace "+" "_" }}
{{ include "pr-review-ai.selectorLabels" . }}
app.kubernetes.io/version: {{ .Values.image.tag | default .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels.
*/}}
{{- define "pr-review-ai.selectorLabels" -}}
app.kubernetes.io/name: {{ include "pr-review-ai.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Database URL: use explicit value if set, otherwise construct from postgresql subchart.
*/}}
{{- define "pr-review-ai.databaseUrl" -}}
{{- if .Values.env.DATABASE_URL }}
{{- .Values.env.DATABASE_URL }}
{{- else }}
{{- printf "postgresql+asyncpg://%s:%s@%s-postgresql:5432/%s" .Values.postgresql.auth.username .Values.postgresql.auth.password .Release.Name .Values.postgresql.auth.database }}
{{- end }}
{{- end }}
