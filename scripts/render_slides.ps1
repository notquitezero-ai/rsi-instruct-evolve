# Export every slide of a .pptx to PNG through the installed PowerPoint (visual QA).
#   .\scripts\render_slides.ps1 deliverables\RSI_演示.pptx deliverables\slides
param([Parameter(Mandatory)][string]$Pptx, [Parameter(Mandatory)][string]$OutDir, [int]$Width = 1600)
$Pptx = (Resolve-Path $Pptx).Path
New-Item -ItemType Directory -Force $OutDir | Out-Null
$OutDir = (Resolve-Path $OutDir).Path
Get-ChildItem $OutDir -Filter 'slide-*.png' -ErrorAction SilentlyContinue | Remove-Item -Force
# COM attaches to an already-running PowerPoint (e.g. the user's own window): never Quit() that one,
# it would close the user's open presentations.  Only quit an instance this script started.
$wasRunning = [bool](Get-Process POWERPNT -ErrorAction SilentlyContinue)
$app = New-Object -ComObject PowerPoint.Application
$pres = $null
try {
    $pres = $app.Presentations.Open($Pptx, $true, $false, $false)  # read-only, no title, no window
    $h = [int]($Width * $pres.PageSetup.SlideHeight / $pres.PageSetup.SlideWidth)
    foreach ($s in $pres.Slides) {
        $f = Join-Path $OutDir ('slide-{0:D2}.png' -f $s.SlideIndex)
        $s.Export($f, 'PNG', $Width, $h)
        $f
    }
} finally {
    if ($pres) { $pres.Close() }   # closes only the copy this script opened
    if (-not $wasRunning) { $app.Quit() }
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($app) | Out-Null
}
