$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$portableRoot = [IO.Path]::GetFullPath((Join-Path $projectRoot 'tools\windows'))
$sevenZip = 'C:\Program Files\7-Zip\7z.exe'
if (-not (Test-Path -LiteralPath $sevenZip)) { throw '7-Zip is required to unpack the signed MSI files.' }
if (-not $portableRoot.StartsWith($projectRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Windows tools must stay inside the project directory.'
}
$guiDefinitions = @(
    @{Name='NsightSystems-2026.6.1'; Executable='nsys-ui.exe'; Shortcut='Nsight Systems.lnk'; Report='env-smoke.nsys-rep'},
    @{Name='NsightCompute-2026.4.0'; Executable='ncu-ui.exe'; Shortcut='Nsight Compute.lnk'; Report='env-smoke.ncu-rep'}
)
$guiResults = foreach ($guiDefinition in $guiDefinitions) {
    $installerPath = Join-Path $projectRoot ('installers\windows\' + $guiDefinition.Name + '.msi')
    $signature = Get-AuthenticodeSignature -LiteralPath $installerPath
    if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch 'NVIDIA Corporation') {
        throw ('Unverified NVIDIA installer: ' + $guiDefinition.Name)
    }
    $destinationPath = Join-Path $portableRoot $guiDefinition.Name
    New-Item -ItemType Directory -Path $destinationPath -Force | Out-Null
    $logPath = Join-Path $projectRoot ('results\environment\' + $guiDefinition.Name + '-unpack.log')
    $stagePath = Join-Path $projectRoot ('.cache\windows-msi\' + $guiDefinition.Name)
    $payloadPath = Join-Path $stagePath 'payload'
    New-Item -ItemType Directory -Path $payloadPath -Force | Out-Null
    # Read MSI layout without executing installation/custom actions. Extract CAB
    # streams and reconstruct only files below INSTALLDIR with long-path support.
    $msiReader = New-Object -ComObject WindowsInstaller.Installer
    $msiDatabase = $msiReader.OpenDatabase($installerPath, 0)
    $directories = @{}
    $directoryQuery = $msiDatabase.OpenView('SELECT `Directory`,`Directory_Parent`,`DefaultDir` FROM `Directory`')
    $null = $directoryQuery.Execute()
    while ($directoryRow = $directoryQuery.Fetch()) {
        $targetName = (($directoryRow.StringData(3) -split ':')[0] -split '\|')[-1]
        $directories[$directoryRow.StringData(1)] = @{Parent=$directoryRow.StringData(2);Name=$targetName}
    }
    if (-not $directories.ContainsKey('INSTALLDIR')) { throw 'MSI has no INSTALLDIR directory.' }
    $relativeDirectories = @{INSTALLDIR=''}
    function Resolve-MsiDirectory([string]$directoryId) {
        if ($relativeDirectories.ContainsKey($directoryId)) { return $relativeDirectories[$directoryId] }
        if (-not $directories.ContainsKey($directoryId) -or $directoryId -eq 'TARGETDIR') { return $null }
        $entry = $directories[$directoryId]
        $parentRelative = Resolve-MsiDirectory $entry.Parent
        if ($null -eq $parentRelative) { return $null }
        $relative = if ($entry.Name -eq '.') { $parentRelative } elseif ($parentRelative -eq '') { $entry.Name } else { Join-Path $parentRelative $entry.Name }
        $relativeDirectories[$directoryId] = $relative
        return $relative
    }
    $componentDirectories = @{}
    $componentQuery = $msiDatabase.OpenView('SELECT `Component`,`Directory_` FROM `Component`')
    $null = $componentQuery.Execute()
    while ($componentRow = $componentQuery.Fetch()) { $componentDirectories[$componentRow.StringData(1)] = $componentRow.StringData(2) }
    & $sevenZip e $installerPath ('-o' + $stagePath) '*.cab' -y -bso0 -bsp0 *> $logPath
    if ($LASTEXITCODE -ne 0) { throw 'Could not extract MSI cabinet streams.' }
    foreach ($cabinet in (Get-ChildItem -LiteralPath $stagePath -Filter '*.cab' -File)) {
        & $sevenZip e $cabinet.FullName ('-o' + $payloadPath) -y -bso0 -bsp0 *>> $logPath
        if ($LASTEXITCODE -ne 0) { throw 'Could not extract cabinet payload.' }
    }
    $fileQuery = $msiDatabase.OpenView('SELECT `File`,`Component_`,`FileName`,`FileSize` FROM `File`')
    $null = $fileQuery.Execute()
    $copiedFiles = 0
    while ($fileRow = $fileQuery.Fetch()) {
        $relativeDirectory = Resolve-MsiDirectory $componentDirectories[$fileRow.StringData(2)]
        if ($null -eq $relativeDirectory) { continue }
        $fileName = ($fileRow.StringData(3) -split '\|')[-1]
        $relativeFile = if ($relativeDirectory -eq '') { $fileName } else { Join-Path $relativeDirectory $fileName }
        $targetFile = [IO.Path]::GetFullPath((Join-Path $destinationPath $relativeFile))
        if (-not $targetFile.StartsWith($destinationPath + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) { throw 'MSI path escaped destination.' }
        $payloadFile = Join-Path $payloadPath $fileRow.StringData(1)
        if (-not (Test-Path -LiteralPath $payloadFile)) { throw ('Missing payload: ' + $fileName) }
        if ((Get-Item -LiteralPath $payloadFile).Length -ne $fileRow.IntegerData(4)) { throw ('Payload size mismatch: ' + $fileName) }
        [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($targetFile)) | Out-Null
        [IO.File]::Copy($payloadFile, $targetFile, $true)
        $copiedFiles++
    }
    $guiExecutable = Get-ChildItem -LiteralPath $destinationPath -Filter $guiDefinition.Executable -File -Recurse | Select-Object -First 1
    if (-not $guiExecutable) { throw ('GUI executable not found: ' + $guiDefinition.Name) }
    $shortcutShell = New-Object -ComObject WScript.Shell
    $shortcut = $shortcutShell.CreateShortcut((Join-Path $projectRoot $guiDefinition.Shortcut))
    $shortcut.TargetPath = $guiExecutable.FullName
    $shortcut.WorkingDirectory = $projectRoot
    $reportPath = Join-Path $projectRoot ('results\profiling\' + $guiDefinition.Report)
    if ($guiDefinition.Report -eq 'env-smoke.ncu-rep' -and (Test-Path -LiteralPath ($reportPath + 'z'))) {
        $reportPath += 'z'
    }
    if (Test-Path -LiteralPath $reportPath) { $shortcut.Arguments = '"' + $reportPath + '"' }
    $shortcut.Save()
    [pscustomobject]@{Name=$guiDefinition.Name;Executable=$guiExecutable.FullName;Version=$guiExecutable.VersionInfo.ProductVersion;Shortcut=$guiDefinition.Shortcut;Files=$copiedFiles;Mode='Project-local unpacked viewer; no system registration'}
}
$guiResults | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $projectRoot 'results\environment\windows-gui.json') -Encoding utf8
$guiResults | ConvertTo-Json -Depth 4
