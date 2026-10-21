# High-resolution screenshot of the running demo (for the PPT), via headless Edge + DevTools protocol.
#   .\scripts\capture_demo.ps1 [-Url http://127.0.0.1:7860/?__theme=light] [-Out deliverables\demo_screenshot.png]
param([string]$Url = 'http://127.0.0.1:7860/?__theme=light', [string]$Out = 'deliverables\demo_screenshot.png',
      [int]$Width = 1440, [int]$Height = 1180, [int]$WaitSec = 15, [int]$Port = 9333)
$root = Split-Path $PSScriptRoot -Parent
$edge = @('C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe', 'C:\Program Files\Microsoft\Edge\Application\msedge.exe') |
    Where-Object { Test-Path $_ } | Select-Object -First 1
$profileDir = Join-Path $root 'tmp\edge-profile'   # keep the browser profile on D:, not in C:\Users
$proc = Start-Process $edge -PassThru -ArgumentList @('--headless=new', '--disable-gpu', '--hide-scrollbars',
    "--remote-debugging-port=$Port", "--user-data-dir=$profileDir", "--window-size=$Width,$Height", 'about:blank')
try {
    $target = $null
    for ($i = 0; $i -lt 30 -and -not $target; $i++) {
        Start-Sleep -Milliseconds 500
        try { $target = (Invoke-RestMethod "http://127.0.0.1:$Port/json") | Where-Object type -eq 'page' | Select-Object -First 1 } catch {}
    }
    $ws = New-Object System.Net.WebSockets.ClientWebSocket
    $ws.Options.KeepAliveInterval = [TimeSpan]::FromSeconds(30)
    $ws.ConnectAsync([Uri]$target.webSocketDebuggerUrl, [Threading.CancellationToken]::None).Wait()
    $script:id = 0
    function Send-Cdp($method, $params) {
        $script:id++
        $msg = @{ id = $script:id; method = $method; params = $params } | ConvertTo-Json -Compress -Depth 5
        $bytes = [Text.Encoding]::UTF8.GetBytes($msg)
        $ws.SendAsync([ArraySegment[byte]]$bytes, 'Text', $true, [Threading.CancellationToken]::None).Wait()
        $buf = New-Object byte[] 1048576
        while ($true) {   # read frames until the reply with our id arrives
            $sb = New-Object Text.StringBuilder
            do {
                $r = $ws.ReceiveAsync([ArraySegment[byte]]$buf, [Threading.CancellationToken]::None).Result
                [void]$sb.Append([Text.Encoding]::UTF8.GetString($buf, 0, $r.Count))
            } while (-not $r.EndOfMessage)
            $obj = $sb.ToString() | ConvertFrom-Json
            if ($obj.id -eq $script:id) { return $obj }
        }
    }
    [void](Send-Cdp 'Emulation.setDeviceMetricsOverride' @{ width = $Width; height = $Height; deviceScaleFactor = 1.5; mobile = $false })
    [void](Send-Cdp 'Page.navigate' @{ url = $Url })
    Start-Sleep $WaitSec   # real time, so the Gradio backend can answer the load events
    $shot = Send-Cdp 'Page.captureScreenshot' @{ format = 'png' }
    $path = Join-Path $root $Out
    [IO.File]::WriteAllBytes($path, [Convert]::FromBase64String($shot.result.data))
    $path
} finally {
    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
}
