param(
    [string]$Destination = 'D:\s_dis\Downloads\ai-infra-wheels\flashinfer-0.7.2.post1-cu130'
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$version = '0.7.2.post1'
$cudaLabel = 'cu130'
New-Item -ItemType Directory -Path $Destination -Force | Out-Null
$release = Invoke-RestMethod -Uri ('https://api.github.com/repos/flashinfer-ai/flashinfer/releases/tags/v' + $version) -TimeoutSec 30
$releaseAssetsByName = @{}
foreach ($asset in $release.assets) { $releaseAssetsByName[$asset.name] = $asset }

function Resolve-OfficialWheel([string]$Distribution, [string]$ExpectedFile, [string]$IndexRoot) {
    $indexUrl = $IndexRoot.TrimEnd('/') + '/' + $Distribution + '/'
    $page = Invoke-WebRequest -Uri $indexUrl -TimeoutSec 60
    foreach ($match in [regex]::Matches($page.Content, 'href=["'']([^"'']+)["'']')) {
        $href = [Net.WebUtility]::HtmlDecode($match.Groups[1].Value)
        $uri = [uri]::new([uri]$indexUrl, $href)
        $name = [uri]::UnescapeDataString(($uri.AbsolutePath -split '/')[-1])
        if ($name -ne $ExpectedFile) { continue }
        if ($uri.Host -notin @('github.com', 'flashinfer.ai')) { throw 'Unexpected wheel host' }
        $hash = $uri.Fragment -replace '^#sha256=', ''
        if ($hash -notmatch '^[0-9a-fA-F]{64}$') { throw ('Missing official SHA-256: ' + $name) }
        return [pscustomobject]@{Name=$name;Url=$uri.GetLeftPart([UriPartial]::Path);SHA256=$hash;Distribution=$Distribution}
    }
    throw ('Exact official wheel not found: ' + $ExpectedFile)
}

function Save-VerifiedWheel($Wheel) {
    $path = Join-Path $Destination $Wheel.Name
    if (Test-Path -LiteralPath $path) {
        if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ine $Wheel.SHA256) {
            throw ('Existing wheel has a different hash: ' + $Wheel.Name)
        }
        Write-Output ('Already verified: ' + $Wheel.Name)
        return
    }
    Write-Output ('Downloading: ' + $Wheel.Name)
    $partial = $path + '.part'
    $asset = $releaseAssetsByName[$Wheel.Name]
    if (-not $asset) { throw ('Official release asset not found: ' + $Wheel.Name) }
    if ($asset.digest -and $asset.digest -ine ('sha256:' + $Wheel.SHA256)) { throw 'Release digest differs from wheel index digest' }
    # The GitHub API asset endpoint works through the existing Windows proxy;
    # direct release URL GETs timed out. No global proxy changes are made.
    $downloadArguments = @{
        Uri=($asset.url + '?download=1')
        Headers=@{Accept='application/octet-stream';'X-GitHub-Api-Version'='2022-11-28'}
        OutFile=$partial
        Resume=$true
        TimeoutSec=180
        MaximumRetryCount=2
        RetryIntervalSec=3
    }
    Invoke-WebRequest @downloadArguments
    if ((Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash -ine $Wheel.SHA256) {
        throw ('Official SHA-256 mismatch: ' + $Wheel.Name)
    }
    Move-Item -LiteralPath $partial -Destination $path -ErrorAction Stop
    Write-Output ('Verified: ' + $Wheel.Name)
}

$shimFile = 'flashinfer_jit_cache-' + $version + '+' + $cudaLabel + '-cp39-abi3-manylinux_2_28_x86_64.whl'
$shim = Resolve-OfficialWheel 'flashinfer-jit-cache' $shimFile 'https://flashinfer.ai/whl/cu130'
Save-VerifiedWheel $shim
$archive = [IO.Compression.ZipFile]::OpenRead((Join-Path $Destination $shim.Name))
try {
    $metadataEntries = @($archive.Entries | Where-Object { $_.FullName -like '*.dist-info/METADATA' })
    if ($metadataEntries.Count -ne 1) { throw 'Unexpected shim metadata layout' }
    $reader = [IO.StreamReader]::new($metadataEntries[0].Open())
    try { $metadata = $reader.ReadToEnd() } finally { $reader.Dispose() }
} finally { $archive.Dispose() }
$providerNames = @([regex]::Matches($metadata, '(?m)^Requires-Dist:\s*(flashinfer-jit-cache-sm[0-9]+[af]?)\s*==\s*0\.7\.2\.post1\+cu130\s*$') | ForEach-Object { $_.Groups[1].Value } | Sort-Object -Unique)
if ($providerNames.Count -lt 1) { throw 'No pinned architecture providers found in official shim metadata' }
$plan = @($shim)
$plan += Resolve-OfficialWheel 'flashinfer-cubin' ('flashinfer_cubin-' + $version + '-py3-none-any.whl') 'https://flashinfer.ai/whl'
foreach ($providerName in $providerNames) {
    $providerFile = ($providerName -replace '-', '_') + '-' + $version + '+' + $cudaLabel + '-cp39-abi3-manylinux_2_28_x86_64.whl'
    $plan += Resolve-OfficialWheel $providerName $providerFile 'https://flashinfer.ai/whl/cu130'
}
$plan | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $projectRoot 'results\environment\flashinfer-wheel-manifest.json') -Encoding utf8
foreach ($wheel in $plan | Select-Object -Skip 1) { Save-VerifiedWheel $wheel }
Write-Output ('All official wheels verified in ' + $Destination)
