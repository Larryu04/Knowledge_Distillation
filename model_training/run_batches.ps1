# Generates 1,200 examples per course for the 11 new domains, in two sequential
# batches (6 then 5 concurrent jobs) to stay under gpt-4o-mini rate limits.
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
Set-Location $root

# 1. Load OPENAI_API_KEY from .env
$envLine = Get-Content (Join-Path $root ".env") | Where-Object { $_ -match '^\s*OPENAI_API_KEY\s*=' } | Select-Object -First 1
if (-not $envLine) { throw "OPENAI_API_KEY not found in .env" }
$env:OPENAI_API_KEY = ($envLine -replace '^\s*OPENAI_API_KEY\s*=\s*', '').Trim().Trim('"').Trim("'")

$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { $python = "python" }
$logDir = Join-Path $root "logs"
New-Item -ItemType Directory -Force $logDir | Out-Null

$commonArgs = @("--num", "1200", "--model", "gpt-4o-mini", "--sleep-between-calls", "2.5")

function Start-CourseJob($course, $output, $extraArgs) {
    $log = Join-Path $logDir ([IO.Path]::GetFileNameWithoutExtension($output) + ".log")
    $pyArgs = @("generate_pilot_full-dataset.py", "--course", $course, "--output", $output) + $commonArgs + $extraArgs
    Write-Host "Starting: $course -> $output (log: $log)"
    Start-Job -Name $course -ArgumentList $root, $python, $pyArgs, $log, $env:OPENAI_API_KEY -ScriptBlock {
        param($root, $python, $pyArgs, $log, $key)
        Set-Location $root
        $env:OPENAI_API_KEY = $key
        $env:PYTHONIOENCODING = "utf-8"
        & $python -u @pyArgs *> $log
        "exit code: $LASTEXITCODE"
    }
}

function Wait-Batch($jobs, $name) {
    Write-Host "Waiting for $name ($($jobs.Count) jobs)..."
    $jobs | Wait-Job | Out-Null
    foreach ($j in $jobs) {
        $result = Receive-Job $j
        Write-Host ("  [{0}] {1} - {2}" -f $j.State, $j.Name, ($result -join ' '))
    }
    $jobs | Remove-Job
    Write-Host "$name finished."
}

# 2-3. Batch 1
$batch1 = @(
    Start-CourseJob "Introduction to AI"               "intro_ai_data.json"       @()
    Start-CourseJob "AI-Based Software Development"    "ai_software_data.json"    @()
    Start-CourseJob "Introduction to Machine Learning" "intro_ml_data.json"       @()
    Start-CourseJob "Deep Learning"                    "deep_learning_data.json"  @()
    Start-CourseJob "Computability and Complexity"     "computability_data.json"  @()
    Start-CourseJob "Introduction to Cryptography"     "cryptography_data.json"   @("--max-consecutive-skips", "20")
)
Wait-Batch $batch1 "Batch 1"

# 4-5. Batch 2
$batch2 = @(
    Start-CourseJob "Information Systems Foundations"    "info_systems_data.json"      @()
    Start-CourseJob "Computational Biology Applications" "comp_biology_data.json"      @()
    Start-CourseJob "Discrete Mathematics"               "discrete_math_data.json"     @()
    Start-CourseJob "Logic and Set Theory"               "logic_set_theory_data.json"  @()
    Start-CourseJob "Graph Theory"                       "graph_theory_data.json"      @()
)
Wait-Batch $batch2 "Batch 2"

Write-Host "All batches complete."
