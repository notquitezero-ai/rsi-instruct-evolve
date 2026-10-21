# Dot-source before any project command:  . .\scripts\env.ps1
# Redirects every cache / temp directory into this project on D: so nothing grows on C:.
$RsiRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$RsiCache = Join-Path $RsiRoot 'cache'
$RsiPaths = @{
    TEMP = (Join-Path $RsiRoot 'tmp')
    TMP = (Join-Path $RsiRoot 'tmp')
    PIP_CACHE_DIR = (Join-Path $RsiCache 'pip')
    HF_HOME = (Join-Path $RsiCache 'huggingface')
    HF_HUB_CACHE = (Join-Path $RsiCache 'huggingface\hub')
    HF_DATASETS_CACHE = (Join-Path $RsiCache 'huggingface\datasets')
    TORCH_HOME = (Join-Path $RsiCache 'torch')
    TRITON_CACHE_DIR = (Join-Path $RsiCache 'triton')
    XDG_CACHE_HOME = (Join-Path $RsiCache 'xdg')
    CUDA_CACHE_PATH = (Join-Path $RsiCache 'cuda')
    GRADIO_TEMP_DIR = (Join-Path $RsiCache 'gradio')
    MPLCONFIGDIR = (Join-Path $RsiCache 'matplotlib')
    PYTHONPYCACHEPREFIX = (Join-Path $RsiCache 'pycache')
    NLTK_DATA = (Join-Path $RsiCache 'nltk_data')   # IFEval's sentence splitter; default would be %APPDATA% on C:
}
foreach ($kv in $RsiPaths.GetEnumerator()) {
    [System.IO.Directory]::CreateDirectory($kv.Value) | Out-Null
    [Environment]::SetEnvironmentVariable($kv.Key, $kv.Value, 'Process')
}
$env:PYTHONNOUSERSITE = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
$env:TOKENIZERS_PARALLELISM = 'false'
$env:HF_HUB_DISABLE_TELEMETRY = '1'
$env:GRADIO_ANALYTICS_ENABLED = 'False'
$env:PYTORCH_CUDA_ALLOC_CONF = 'expandable_segments:True'
$RsiPython = Join-Path $RsiRoot '.venv\Scripts\python.exe'
Remove-Variable kv, RsiPaths, RsiCache
